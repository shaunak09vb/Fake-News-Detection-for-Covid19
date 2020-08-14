from flask import Flask, request, render_template
import pickle


app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify',methods=['POST'])
def classify():
    result=request.form
    query_title=result['news_title']
    query_content=result['news_content']
    total=query_title+query_content
    data=[total]
    pred=model.predict(data)
    return render_template('index.html', prediction_text='The news is : {}'.format(pred))
       
    
    
    
if __name__=="__main__":
    app.run(debug=True)
    