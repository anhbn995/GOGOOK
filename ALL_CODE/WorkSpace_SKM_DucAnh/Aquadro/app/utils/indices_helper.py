SENTINEL1 = 'Sentinel1'
SENTINEL2 = 'Sentinel2'
PLANET = 'Planet'

INDICES = {
    'NDVI': {
        'source': SENTINEL2,
        'expression': "(B08-B04)/(B08+B04)",
        'statistics': [
            {'min_value': 0, 'color': '#F44336', 'label': 'Stressed (0-0.2)'},
            {'min_value': 0.2, 'color': '#FF9800', 'label': 'Slightly Stressed (0.2-0.4)'},
            {'min_value': 0.4, 'color': '#FFEB3B', 'label': 'Moderate (0.4-0.6)'},
            {'min_value': 0.6, 'color': '#8BC34A', 'label': 'Healthy (0.6-0.8)'},
            {'min_value': 0.8, 'color': '#4CAF50', 'label': 'Very Healthy (0.8-1)'}
        ]
    },
    'SAVI': {
        'source': SENTINEL2,
        'expression': "1.5*(B08-B04)/(B08+B04+0.5)",
        'statistics': [
            {'min_value': 0, 'color': '#800000', 'label': 'Open (0-0.2)'},
            {'min_value': 0.2, 'color': '#F44336', 'label': 'Low Dense (0.2-0.4)'},
            {'min_value': 0.4, 'color': '#FF9800', 'label': 'Slightly Dense (0.4-0.6)'},
            {'min_value': 0.6, 'color': '#FFEB3B', 'label': 'Dense (0.6-0.8)'},
            {'min_value': 0.8, 'color': '#8BC34A', 'label': 'Highly Dense (0.8-1)'},
            {'min_value': 1, 'color': '#4CAF50', 'label': 'Very High (1+)'}
        ]
    },
    'RECI': {
        'source': SENTINEL2,
        'expression': "where(B04 != 0, B08/B04-1, 0);",
        'statistics': [
            {'min_value': 0, 'color': '#F44336', 'label': 'Very Low (0-1)'},
            {'min_value': 1, 'color': '#FF9800', 'label': 'Low (1-1.5)'},
            {'min_value': 1.5, 'color': '#FFEB3B', 'label': 'Normal (1.5-2)'},
            {'min_value': 2, 'color': '#8BC34A', 'label': 'High (2-2.5)'},
            {'min_value': 2.5, 'color': '#4CAF50', 'label': 'Very High (2.5+)'}
        ]
    },
    'GCI': {
        'source': SENTINEL2,
        'expression': "where(B03 != 0, B08/B03-1, 0);",
        'statistics': [
            {'min_value': 0, 'color': '#F44336', 'label': 'Very Low (0-1)'},
            {'min_value': 1, 'color': '#FF9800', 'label': 'Low (1-1.5)'},
            {'min_value': 1.5, 'color': '#FFEB3B', 'label': 'Normal (1.5-2)'},
            {'min_value': 2, 'color': '#8BC34A', 'label': 'High (2-2.5)'},
            {'min_value': 2.5, 'color': '#4CAF50', 'label': 'Very High (2.5+)'}
        ]
    },
    'EVI2': {
        'source': SENTINEL2,
        'expression': "2.5*(B08-B04)/(B08+2.4*B04+1)",
        'statistics': [
            {'min_value': 0, 'color': '#800000', 'label': 'Extremely Low (0-0.2)'},
            {'min_value': 0.2, 'color': '#F44336', 'label': 'Very Low (0.2-0.5)'},
            {'min_value': 0.5, 'color': '#FF9800', 'label': 'Low (0.5-0.8)'},
            {'min_value': 0.8, 'color': '#FFEB3B', 'label': 'Normal (0.8-1.1)'},
            {'min_value': 1.1, 'color': '#8BC34A', 'label': 'High (1.1-1.4)'},
            {'min_value': 1.4, 'color': '#4CAF50', 'label': 'Very High (1.4+)'}
        ]
    },
    'SIPI': {
        'source': SENTINEL2,
        'expression': "where((B08-B04) != 0, (B08-B02)/(B08-B04), 0);",
        'statistics': [
            {'min_value': 0.8, 'color': '#FFEB3B', 'label': 'Stressed (0.8-1)'},
            {'min_value': 1, 'color': '#8BC34A', 'label': 'Healthy (1-1.2)'},
            {'min_value': 1.2, 'color': '#4CAF50', 'label': 'Very Healthy (1.2+)'}
        ]
    },
    'NDRE': {
        'source': SENTINEL2,
        'expression': " (B08-B05)/(B08+B05)",
        'statistics': [
            {'min_value': 0, 'color': '#F44336', 'label': 'Open Soil (0-0.2)'},
            {'min_value': 0.2, 'color': '#FFEB3B', 'label': 'Unhealthy (0.2-0.6)'},
            {'min_value': 0.6, 'color': '#8BC34A', 'label': 'Healthy (0.6-1)'}
        ]
    },
    'NDMI': {
        'source': SENTINEL2,
        'expression': "(B08-B11)/(B08+B11)",
        'statistics': [
            {'min_value': 0, 'color': '#F44336', 'label': 'Average (0-0.2)'},
            {'min_value': 0.2, 'color': '#FF9800', 'label': 'Mid-high (0.2-0.4)'},
            {'min_value': 0.4, 'color': '#FFEB3B', 'label': 'High (0.4-0.6)'},
            {'min_value': 0.6, 'color': '#8BC34A', 'label': 'Very high (0.6-0.8)'},
            {'min_value': 0.8, 'color': '#4CAF50', 'label': 'Total (0.8-1)'}
        ]
    },
    'NDRVI': {
        'source': SENTINEL1,
        'expression': "2*(vh-vv)/(vh+vv)",
        'statistics': [
            {'min_value': 0, 'color': '#F44336', 'label': 'Stressed (0-0.2)'},
            {'min_value': 0.2, 'color': '#FF9800', 'label': 'Slightly Stressed (0.2-0.4)'},
            {'min_value': 0.4, 'color': '#FFEB3B', 'label': 'Moderate (0.4-0.6)'},
            {'min_value': 0.6, 'color': '#8BC34A', 'label': 'Healthy (0.6-0.8)'},
            {'min_value': 0.8, 'color': '#4CAF50', 'label': 'Very Healthy (0.8-1)'}
        ]
    },
    'RI': {
        'source': SENTINEL1,
        'expression': "where(vv != 0, vh/vv, 0);",
        'statistics': [
            {'min_value': 0, 'color': '#F44336', 'label': 'Very Low (0-1)'},
            {'min_value': 1, 'color': '#FF9800', 'label': 'Low (1-1.25)'},
            {'min_value': 1.25, 'color': '#FFEB3B', 'label': 'Normal (1.25-1.5)'},
            {'min_value': 1.5, 'color': '#8BC34A', 'label': 'High (1.5-1.75)'},
            {'min_value': 1.75, 'color': '#4CAF50', 'label': 'Very High (1.75+)'}
        ]
    },
    'RVI': {
        'source': SENTINEL1,
        'expression': "4*vh/(vh+vv)",
        'statistics': [
            {'min_value': 0, 'color': '#F44336', 'label': 'Background (0-0.5)'},
            {'min_value': 0.5, 'color': '#FF9800', 'label': 'Very Early (0.5-1)'},
            {'min_value': 1, 'color': '#FFEB3B', 'label': 'Early (1-1.5)'},
            {'min_value': 1.5, 'color': '#8BC34A', 'label': 'Timely (1.5-2.5)'},
            {'min_value': 2.5, 'color': '#4CAF50', 'label': 'Slightly Late (2.5+)'}
        ]
    },
    'RVI4S1': {
        'source': SENTINEL1,
        'expression': "sqrt(vv/(vv+vh))*4*vh/(vh+vv)",
        'statistics': [
            {'min_value': 0, 'color': '#F44336', 'label': 'Open (0-1)'},
            {'min_value': 1, 'color': '#FF9800', 'label': 'Low Dense (1-1.25)'},
            {'min_value': 1.25, 'color': '#FFEB3B', 'label': 'Slightly Dense (1.25-1.5)'},
            {'min_value': 1.5, 'color': '#8BC34A', 'label': 'Dense (1.5-1.75)'},
            {'min_value': 1.75, 'color': '#4CAF50', 'label': 'Highly Dense (1.75+)'}
        ]
    }
}

PLANET_INDICES = {
    'NDVI': {
        'source': 'PSScene',
        'expression': "(b8-b6)/(b8+b6)",
        'statistics': [
            {'min_value': 0, 'color': '#F44336', 'label': 'Stressed (0-0.2)'},
            {'min_value': 0.2, 'color': '#FF9800', 'label': 'Slightly Stressed (0.2-0.4)'},
            {'min_value': 0.4, 'color': '#FFEB3B', 'label': 'Moderate (0.4-0.6)'},
            {'min_value': 0.6, 'color': '#8BC34A', 'label': 'Healthy (0.6-0.8)'},
            {'min_value': 0.8, 'color': '#4CAF50', 'label': 'Very Healthy (0.8-1)'}
        ]
    },
    'SAVI': {
        'source': 'PSScene',
        'expression': "1.5*(b8-b6)/(b8+b6+0.5)",
        'statistics': [
            {'min_value': 0, 'color': '#800000', 'label': 'Open (0-0.2)'},
            {'min_value': 0.2, 'color': '#F44336', 'label': 'Low Dense (0.2-0.4)'},
            {'min_value': 0.4, 'color': '#FF9800', 'label': 'Slightly Dense (0.4-0.6)'},
            {'min_value': 0.6, 'color': '#FFEB3B', 'label': 'Dense (0.6-0.8)'},
            {'min_value': 0.8, 'color': '#8BC34A', 'label': 'Highly Dense (0.8-1)'},
            {'min_value': 1, 'color': '#4CAF50', 'label': 'Very High (1+)'}
        ]
    },
    'RECI': {
        'source': 'PSScene',
        'expression': "where(b6 != 0, b8/b6-1, 0);",
        'statistics': [
            {'min_value': 0, 'color': '#F44336', 'label': 'Very Low (0-1)'},
            {'min_value': 1, 'color': '#FF9800', 'label': 'Low (1-1.5)'},
            {'min_value': 1.5, 'color': '#FFEB3B', 'label': 'Normal (1.5-2)'},
            {'min_value': 2, 'color': '#8BC34A', 'label': 'High (2-2.5)'},
            {'min_value': 2.5, 'color': '#4CAF50', 'label': 'Very High (2.5+)'}
        ]
    },
    'GCI': {
        'source': 'PSScene',
        'expression': "where(b4 != 0, b8/b4-1, 0);",
        'statistics': [
            {'min_value': 0, 'color': '#F44336', 'label': 'Very Low (0-1)'},
            {'min_value': 1, 'color': '#FF9800', 'label': 'Low (1-1.5)'},
            {'min_value': 1.5, 'color': '#FFEB3B', 'label': 'Normal (1.5-2)'},
            {'min_value': 2, 'color': '#8BC34A', 'label': 'High (2-2.5)'},
            {'min_value': 2.5, 'color': '#4CAF50', 'label': 'Very High (2.5+)'}
        ]
    },
    'EVI2': {
        'source': 'PSScene',
        'expression': "2.5*(b8-b6)/(b8+2.4*b6+1)",
        'statistics': [
            {'min_value': 0, 'color': '#800000', 'label': 'Extremely Low (0-0.2)'},
            {'min_value': 0.2, 'color': '#F44336', 'label': 'Very Low (0.2-0.5)'},
            {'min_value': 0.5, 'color': '#FF9800', 'label': 'Low (0.5-0.8)'},
            {'min_value': 0.8, 'color': '#FFEB3B', 'label': 'Normal (0.8-1.1)'},
            {'min_value': 1.1, 'color': '#8BC34A', 'label': 'High (1.1-1.4)'},
            {'min_value': 1.4, 'color': '#4CAF50', 'label': 'Very High (1.4+)'}
        ]
    },
    'SIPI': {
        'source': 'PSScene',
        'expression': "where((b8-b6) != 0, (b8-b2)/(b8-b6), 0);",
        'statistics': [
            {'min_value': 0.8, 'color': '#FFEB3B', 'label': 'Stressed (0.8-1)'},
            {'min_value': 1, 'color': '#8BC34A', 'label': 'Healthy (1-1.2)'},
            {'min_value': 1.2, 'color': '#4CAF50', 'label': 'Very Healthy (1.2+)'}
        ]
    },
    'NDRE': {
        'source': 'PSScene',
        'expression': " (b8-b7)/(b8+b7)",
        'statistics': [
            {'min_value': 0, 'color': '#F44336', 'label': 'Open Soil (0-0.2)'},
            {'min_value': 0.2, 'color': '#FFEB3B', 'label': 'Unhealthy (0.2-0.6)'},
            {'min_value': 0.6, 'color': '#8BC34A', 'label': 'Healthy (0.6-1)'}
        ]
    }
}

def index_expression(index):
    return INDICES[index]['expression']


def index_colors(index):
    return [(item['min_value'], item['color']) for item in INDICES[index]['statistics']]


def index_statistics(index):
    return INDICES[index]['statistics']


def index_source(index):
    return INDICES[index]['source']
