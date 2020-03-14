
from flask import Flask, render_template ,request
from sklearn.feature_extraction.text import CountVectorizer
from joblib import load
import pickle

app = Flask(__name__)

@app.route('/')
def page1():
   model1 = load('toxic_model.jbl') 
   model2 = load('severe_toxic_model.jbl') 
   model3 = load('obscene_model.jbl') 
   model4 = load('threat_model.jbl') 
   model5 = load('insult_model.jbl') 
   model6 = load('identity_hate_model.jbl') 
   model_list = [model1,model2,model3,model4,model5,model6]
   return render_template('forms.html',l = model_list)

@app.route('/output',methods = ['POST', 'GET'])
def page2():
    if request.method == 'POST':
      model1 = load('toxic_model.jbl') 
      model2 = load('severe_toxic_model.jbl') 
      model3 = load('obscene_model.jbl') 
      model4 = load('threat_model.jbl') 
      model5 = load('insult_model.jbl') 
      model6 = load('identity_hate_model.jbl') 
      model_list = [model1,model2,model3,model4,model5,model6]

      output = request.form
      comment = output.items()
      app.logger.info(str(output['comment']))
      s = str(output['comment'])

      cv = pickle.load(open("cv_vector.pickle", "rb"))
      X = cv.transform([s])
      y = [  i.predict(X) for i in model_list]  
      y = [ list(i)[0] for i in y]
      app.logger.info(str(y))
      string = 'toxic: {} severe_toxic: {} obscene: {} threat:{} insult: {} identiy_hate: {}'.format(y[0],y[1],y[2],y[3],y[4],y[5])
      app.logger.info(string)

      return render_template("forms1.html", output = output , y_pred =string)

if __name__ == '__main__':
   app.run(debug=True)