from flask import Flask
app = Flask(__name__)#referance the file
@app.route("/home/<string:name><int:id>")

def hello(name , id):
    return {"Hello World!": name , "id":id}

if __name__ == "__main__":
    app.run(debug=True)