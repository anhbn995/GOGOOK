from flask import Flask,render_template,url_for,request,redirect
from detecto import core, utils, visualize
import os
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {'png', 'jfif', 'mov', 'jpg', 'jpeg', 'gif'}
app=Flask(__name__)


@app.route('/')
def game1():
  return render_template('home.html') 
@app.route('/document')

def game():
  return render_template('index.html') 


@app.route('/implement')
def gam():
  return render_template('base.html')
  
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file'] 
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(basepath,secure_filename(f.filename))
        
       
        
        
        f.save(file_path)
        model = core.Model.load('model_weights.pth', ['tiger', 'elephant', 'panda'])
        image = utils.read_image(file_path)
        predictions = model.predict(image)
        
        labels, boxes, scores = predictions
        
        scores=scores
        
        alt_score=[]
        for i in scores:
            alt_score.append(float(i))
        
        ele=[0]
        tig=[0]
        pan=[0]
        j=0
        for i in labels:
            if i=="elephant":
                ele.append(alt_score[j])
            elif i=="tiger":
                tig.append(alt_score[j])
            elif i=="panda":
                pan.append(alt_score[j])
            j=j+1
        final=[]    
        elephant_score=max(ele)
        tiger_score=max(tig)
        panda_score=max(pan)
        
        
        
        
        
        
        elephant_score=round(elephant_score*100,2)
        tiger_score=round(tiger_score*100,2)
        panda_score=round(panda_score*100,2)
        if (elephant_score>75):
            final.append("Voi")
        if(tiger_score>75):
            final.append("Hổ")
        if(panda_score>75):
            final.append("Gấu Trúc")
        
        
        
       
        return render_template("base.html",elephant_score=elephant_score,tiger_score=tiger_score,panda_score=panda_score,final=final,len=len(final))
        final=[]



if __name__=='__main__':
     app.run(debug=True)