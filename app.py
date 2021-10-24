from flask import Flask, render_template, request, redirect, url_for
import os
import glob
from method import methode


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route("/")
def index():
    return render_template("index.html")

#Handling error 500 and displaying relevant web page
@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html'),500

@app.route('/', methods=['POST'])
def upload_files():
    print("here")
    if request.method == 'POST':
        if (request.files):
            files = request.files.getlist("images")
            print(files)
            for file in files:
                arr_name = file.filename.split(".")
                type_file = arr_name[len(arr_name) - 1]
                file.save(os.path.join(
                    app.config['UPLOAD_FOLDER'], file.filename))
                location = (os.path.join(
                    app.config['UPLOAD_FOLDER'], file.filename))
    result = {}
    status = methode(location)
    print(status)
    result["risk"] = status["risk"]
    result["status"] = status["recomend"]
    result["scroller"] = "result"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run()
