import logging.config

from flask import Flask
from flask import render_template, request, redirect, url_for
from src.train import get_model, predict_ind
from src.createdb import Transaction, ResponseManager

# Initialize Flask app
app = Flask(__name__, template_folder="app/templates", static_folder="app/static")

# Configure flask app from flask_config.py
app.config.from_pyfile("config/flaskconfig.py")

# Define LOGGING_CONFIG from flask_config.py
logging.config.fileConfig(app.config["LOGGING_CONFIG"])
logger = logging.getLogger(app.config["APP_NAME"])

# Load model and encoder to make new predictions
model_path = app.config["MODEL_PATH"]
encoder_path = app.config["ENCODER_PATH"]
scaler_path = app.config["SCALER_PATH"]
model, enc, scaler = get_model(model_path, encoder_path, scaler_path)

# Manager to query data from sql table
response_manager = ResponseManager(app)

@app.route("/", methods=["GET", "POST"])
def home():
    '''Main page of application providing information and collecting form info
    Args:
        None
    Returns:
        rendered html template
    '''
    if request.method == "GET":
        try:
            logger.info("Main page returned")
            return render_template("index.html")
        except Exception as error:
            logger.error("Error page returned with error: %s", error)
            return render_template("error.html")

    if request.method == "POST":
        try:
            representative = str(request.form["representative"])
            ticker = str(request.form["ticker"])
            owner = str(request.form["owner"])
            type_trans = str(request.form["type"])
            amount = str(request.form["amount"])
            cat_vars = [owner, ticker, type_trans, amount, representative]
            trans_price = float(request.form["trans_price"])
            prediction = predict_ind(model, enc, scaler, cat_vars, trans_price)
            url_for_post = url_for("response_page", class1 = str(representative), prob1=prediction)
            logger.info("Prediction submitted from form")
            return redirect(url_for_post)
        except Exception as error:
            logger.error("Error page returned with error: %s", error)
            return render_template("error.html")

@app.route("/response.html/<class1>/<prob1>",
            methods=["GET", "POST"])
def response_page(class1, prob1):
    '''Page that displays model predictions and sql table with additional info
    Args:
        class1 (str): number indicating predicted reason number one
        prob1 (str): probability for predicted reason number one
    Returns:
        rendered html template
    '''
    if request.method == "GET":
        try:
            response = response_manager.session.query(Transaction)\
                                       .filter(Transaction.representative.in_([str(class1)]))
            probs = [prob1]
            logger.info("Response page requested")
            return render_template("response.html", responses = response ,probabilities=probs)
        except Exception as error:
            logger.error("Error getting page: %s", error)
            logger.debug("Make sure to fill entire form")
            return render_template("error.html")

    if request.method == "POST":
        url_for_post = url_for("home/")
        return redirect(url_for_post)

if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"], host=app.config["HOST"])
