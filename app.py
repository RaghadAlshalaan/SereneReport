from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
from PatientReport import reportP
from DailyReport import reportD
from DoctorReport import reportDoc
from PatientReportCustomDuration import reportPC
app = Flask(__name__)  # referance the file
CORS(app)
@app.route("/home/<string:name><int:id>", methods=['GET'])
def hello(name, id):
    return jsonify({"Hello World!": name, "id": id})


@app.route("/patient_report/<string:pid>/<int:du>", methods=['GET'])
def patient_report(pid, du):
    patient_report = reportP(pid, du)
    return jsonify(patient_report)

@app.route("/patient_report_custom_duration/<string:pid>/<string:start>/<string:end>", methods=['GET'])
def patient_custom_report(pid, start , end):
    patient_custom_report = reportPC(pid, start , end)
    return jsonify(patient_custom_report)


@app.route("/daily_report/<string:pid>", methods=['GET'])
def daily_report(pid):
    daily_report = reportD(pid)
    return jsonify(daily_report)


@app.route("/doctor_report/<string:userid>/<string:doctorid>", methods=['GET'])
def doctor_report(userid, doctorid):
    doctor_report = reportDoc(userid, doctorid)
    return jsonify(daily_report)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
