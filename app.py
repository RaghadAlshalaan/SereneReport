from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
from PatientReport import reportP
from DailyReport import reportD
from DoctorReport import doctor
from datetime import timedelta
import datetime
from datetime import date


app = Flask(__name__)  # referance the file
CORS(app)
@app.route("/doctor_report/<string:userid>/<string:doctorid>", methods=['GET'])
def doctor_report(userid, doctorid):
    doctor_report = doctor(userid, doctorid)
    return jsonify(doctor_report)

@app.route("/home/<string:name>/<int:id>", methods=['GET'])
def hello(name, id):
    return jsonify({"Hello World!": name, "id": id})


@app.route("/patient_report/<string:pid>/<int:du>/<string:is_true>", methods=['GET'])
def patient_report(pid, du ,is_true):
    dates =[]
    duration = du
    today=date.today() 
    for x in range(0 ,duration):
        start_date = (today-timedelta(days=duration-x)).isoformat()
        dates.append(start_date)
    patient_report = reportP(pid, dates ,is_true)
    return patient_report

@app.route("/patient_report_custom_duration/<string:pid>/<string:start>/<string:end>/<string:is_true>", methods=['GET'])
def patient_custom_report(pid, start , end ,is_true):
    start_date = datetime.datetime.strptime(start, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end, '%Y-%m-%d').date()

    delta = end_date - start_date
    dates =[]
    for i in range(delta.days + 1):
        day = (start_date + timedelta(days=i)).isoformat()
        dates.append(day)
    patient_custom_report = reportP(pid, dates, is_true)
    return patient_custom_report


@app.route("/daily_report/<string:pid>", methods=['GET'])
def daily_report(pid):
    daily_report = reportD(pid)
    return jsonify(daily_report)

@app.route("/home1/<string:name>/<int:id>", methods=['GET'])
def hello1(name, id):
    return jsonify({"Hello World!": name, "id": id})

@app.route("/home2/<string:name>/<int:id>", methods=['GET'])
def hello2(name, id):
    return jsonify({"Hello World!": name, "id": id})

@app.route("/home3/<string:name>/<int:id>", methods=['GET'])
def hello3(name, id):
    return jsonify({"Hello World!": name, "id": id})




if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False)
