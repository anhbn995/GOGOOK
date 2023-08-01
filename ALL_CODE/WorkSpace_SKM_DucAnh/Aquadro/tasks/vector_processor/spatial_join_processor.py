from tasks.vector_processor import VectorProcessor
from services.vector.postgis import read_postgis
from services.db import engine


class SpatialJoinProcessor(VectorProcessor):
    def __init__(self, task_id: int, token: str, group: str, target_id: int, payload: dict):
        super().__init__(task_id, token, group, target_id, payload)
        self.table_schema = payload.get("table_schema")
        self.join_vector_dataset_table_name = f'"{payload.get("table_schema")}"."{payload.get("join_vector_dataset_table_name")}"'
        self.target_vector_dataset_table_name = f'"{payload.get("table_schema")}"."{payload.get("target_vector_dataset_table_name")}"'
        self.join_type = payload.get('join_type')
        self.join_operation = payload.get('join_operation')
        self.match_option = payload.get('match_option')
        self.fields = payload.get('fields')
        self.output_table_name = self.file_id
        self.result_table = payload.get('result_table')
        self.output_table_schema = 'dynamic_query'
        self.combining_filter_join = payload.get(
            'combining_filter_join') or 'all'
        self.combining_filter_target = payload.get(
            'combining_filter_target') or 'all'
        self.join_type = payload.get('join_type') or 'INNER JOIN'
        self.filters = payload.get('filters') or []

    def prepare_tmp_table(self):
        with engine.connect() as con:
            target_table_schema, target_table_name = self.target_vector_dataset_table_name.replace(
                '"', '').split('.')
            join_table_schema, join_table_name = self.join_vector_dataset_table_name.replace(
                '"', '').split('.')
            target_srid = con.execute(
                f"SELECT Find_SRID('{target_table_schema}','{target_table_name}','geometry')").fetchone()[0]

            target_table_columns = list({f'"{field["column"]}"'
                                         for field in self.fields if field['table'] == 't'})
            join_table_columns = list({f'"{field["column"]}"'
                                       for field in self.fields if field['table'] == 'j'})
            tmp_target_table_name = f"tmp_{target_table_name}"
            tmp_join_table_name = f"tmp_{join_table_name}"
            con.execute(
                f'DROP TABLE IF EXISTS "{target_table_schema}"."{tmp_target_table_name}"')
            con.execute(
                f'''
                CREATE TABLE "{target_table_schema}"."{tmp_target_table_name}" as
                SELECT {','.join(target_table_columns+[''])} geometry, id from {self.target_vector_dataset_table_name}
                {self.create_filters_query('t', self.combining_filter_target)}
                '''
            )
            con.execute(
                f'ALTER TABLE "{target_table_schema}"."{tmp_target_table_name}" ADD PRIMARY KEY (id)')
            con.execute(
                f'CREATE INDEX {target_table_schema}_{tmp_target_table_name}_geometry_idx ON "{target_table_schema}"."{tmp_target_table_name}" USING GIST (geometry)')
            con.execute(
                f'DROP TABLE IF EXISTS "{join_table_schema}"."{tmp_join_table_name}"')
            con.execute(
                f'''
                CREATE TABLE "{join_table_schema}"."{tmp_join_table_name}" as
                SELECT {','.join(join_table_columns+[''])} st_transform(geometry, {target_srid}) as geometry, id from {self.join_vector_dataset_table_name}
                {self.create_filters_query('j', self.combining_filter_join)}
                '''
            )
            con.execute(
                f'ALTER TABLE "{join_table_schema}"."{tmp_join_table_name}" ADD PRIMARY KEY (id)')
            con.execute(
                f'CREATE INDEX {join_table_schema}_{tmp_join_table_name}_geometry_idx ON "{join_table_schema}"."{tmp_join_table_name}" USING GIST (geometry)')

            self.join_vector_dataset_table_name = f'"{join_table_schema}"."{tmp_join_table_name}"'
            self.target_vector_dataset_table_name = f'"{target_table_schema}"."{tmp_target_table_name}"'

    def run_task(self):
        print(self.create_query())
        self.prepare_tmp_table()
        with engine.connect() as con:
            con.execute(
                f'DROP TABLE IF EXISTS "{self.output_table_schema}"."{self.result_table}"')
            con.execute(self.create_query())

    def complete_task(self):
        pass

    def on_success(self):
        print({
            "result_table": self.output_table_name,
            "tile_url": self.tile_url,
            'bbox': self.calculate_bbox(),
        })
        self.store_result(
            {
                "result_table": self.output_table_name,
                "tile_url": self.tile_url,
                'bbox': self.calculate_bbox(),
            }
        )

    def create_query(self):
        join_data_select_str = "t.id as target_id, t.geometry as geometry"
        output_select_str = "target_id"
        group_by_select_str = 'target_id'
        output_str = ''
        row_number = ''
        for field in self.fields:
            alias = ''
            column = ''
            if field['table'] == 't':
                alias = 't'
                column = f"\"{field['name']}\""
                group_by_select_str += ', ' + column
            else:
                alias = 'j'
                column = self.get_aggregate(
                    field['merge_rule'], field['name'])
            join_data_select_str += f", {alias}.\"{field['column']}\" as \"{field['name']}\""
            output_select_str += ', ' + column
        if self.join_operation == 'JOIN_ONE_TO_ONE':
            output_str = f'select {output_select_str}, geometry from join_data jd group by {group_by_select_str}, geometry'
            row_number = ', ROW_NUMBER () OVER (ORDER BY j.id)'
        else:
            output_str = 'select * from join_data'
            row_number = ''
        return f'''\
                CREATE TABLE "{self.output_table_schema}"."{self.output_table_name}" as with join_data as
                (select {join_data_select_str}{row_number}
                from {self.target_vector_dataset_table_name} t {self.join_type} {self.join_vector_dataset_table_name} j
                on ST_{self.match_option}(t.geometry, j.geometry))
                {output_str}'''

    def get_aggregate(self, merge_rule, column):
        aggregate_selections = {
            "First": f"(select \"{column}\" from join_data where target_id = jd.target_id order by row_number asc limit 1) as \"{column}\"",
            "Last": f"(select \"{column}\" from join_data where target_id = jd.target_id order by row_number desc limit 1) as \"{column}\"",
            "Min": f"min(\"{column}\") as \"{column}\"",
            "Max": f"max(\"{column}\") as \"{column}\"",
            "Average": f"avg(\"{column}\") as \"{column}\"",
            "Sum": f"sum(\"{column}\") as \"{column}\"",
            "Count": f"count(\"{column}\") as \"{column}\"",
            "Join": f"string_agg(\"{column}\") as \"{column}\"",
            "Standard deviation": f"stddev(\"{column}\") as \"{column}\"",
        }
        return aggregate_selections.get(merge_rule)

    def create_filters_query(self, table, combining_filter):
        filters = [f for f in self.filters if f['table'] == table]

        if not len(filters):
            return ''
        conbining_operator = ' or ' if combining_filter == 'any' else ' and '
        filter_list = []
        for filter in filters:
            if filter['operator'] == '==':
                filter['operator'] = '='
            if filter['operator'] == '!=':
                filter['operator'] = '<>'
            if filter['operator'] in ['in', 'not in']:
                values = [str(x) if filter['type'] in ['integer', 'double', 'float']
                          else "'"+x.replace("'", "''")+"'" for x in filter['value']]
                filter_value = (', ').join(values)
            else:
                filter['value'] = str(filter['value']).replace("'", "''")
                filter_value = filter['value'] if filter[
                    'type'] in ['integer', 'double', 'float'] else f"'{filter['value']}'"
            filter_value = f'({filter_value})'
            filter_str = f"\"{filter['column']}\" {filter['operator']} {filter_value}"
            if combining_filter == 'none':
                filter_str = 'not ' + filter_str
            filter_list.append(filter_str)
        return f' where {conbining_operator.join(filter_list)}'
