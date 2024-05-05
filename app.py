import joblib
from flask import Flask, request, app, render_template
import numpy as np
import pandas as pd
import csv

app = Flask(__name__)

# Load the model
model = joblib.load(open("model1.joblib", "rb"))

def read_csv_and_create_dict():
    college_dict = {}  # Initialize an empty dictionary to store key-value pairs
    with open("colleges.csv", 'r') as file:
        csv_reader = csv.DictReader(file, delimiter=',')  # Use ',' as delimiter
        
        for row in csv_reader:
            key = int(row['key'])
            college_name = row['college_name']
            college_dict[key] = college_name  # Add key-value pair to dictionary
    
    return college_dict

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    College_code = read_csv_and_create_dict()
    #print(College_code)
    #print(College_code)
    
    Branch_code = {"0": "Computer Science Engineering",
    "1": "Electronics and Communication Engineering",
    "2": "Chemical Engineering",
    "3": "Electrical and Electronics Engineering",
    "4": "Industrial Engineering",
    "5": "Electronics and Instrumentation Engineering",
    "6": "Civil Engineering",
    "7": "Biotechnology",
    "8": "Geoinformatics",
    "9": "Mechanical Engineering",
    "10": "Instrumentation Engineering",
    "11": "Production Engineering",
    "12": "Materials Science and Engineering",
    "13": "Electronics and Computer Engineering",
    "14": "Mining Engineering",
    "15": "Manufacturing Engineering",
    "16": "Artificial Intelligence and Data Science",
    "17": "Chemistry",
    "18": "Industrial BioTechnology",
    "19": "Ceramic Technology",
    "20": "Food Technology",
    "21": "Information Science and Technology",
    "22": "Pharmaceutical Technology",
    "23": "Chemical and Electrochemical Engineering",
    "24": "Computer Science and Engineering",
    "25": "Leather Technology",
    "26": "Artificial Intelligence and Data Science",
    "27": "Textile Technology",
    "28": "Bio-Pharmaceutical Technology",
    "29": "Automobile Engineering",
    "30": "Electrical and Electronic Engineering",
    "31": "Electronics and Instrumentation Engineering",
    "32": "Robotics and Automation",
    "33": "Automobile Engineering (SS)",
    "34": "Production Engineering (SS)",
    "35": "Information Technology",
    "36": "Automobile Engineering (SS)",
    "37": "Computer Science and Engineering (SS)",
    "38": "Information Technology",
    "39": "Electronics and Communication Engineering",
    "40": "Artificial Intelligence and Data Science",
    "41": "Biotechnology (SS)",
    "42": "Soil Mechanics and Foundation Engineering",
    "43": "Construction Engineering and Management",
    "44": "Architectural Engineering",
    "45": "Environmental Engineering",
    "46": "Electrical and Electronics Engineering (SS)",
    "47": "Construction Technology and Management",
    "48": "Bio-Medical Engineering",
    "49": "Fashion Technology and Costume Design",
    "50": "Remote Sensing and GeoInformatics",
    "51": "Mechatronics Engineering",
    "52": "Civil Infrastructure Engineering",
    "53": "Medical Electronics",
    "54": "Agricultural and Irrigation Engineering",
    "55": "Industrial Metallurgy",
    "56": "Automobile Engineering (SS)",
    "57": "Chemical and Electrochemical Engineering (SS)",
    "58": "Petrochemical Technology",
    "59": "Electronics and Communication Engineering",
    "60": "Pharmaceutical Technology (SS)",
    "61": "Electronics and Communication Engineering",
    "62": "Cyber Security",
    "63": "Computer Science and Game Development",
    "64": "Information Technology (SS)",
    "65": "Biomedical and Tissue Engineering",
    "66": "Manufacturing Engineering (SS)",
    "67": "Embedded Systems",
    "68": "Automotive Systems",
    "69": "Food Technology (SS)",
    "70": "Computer Science and Business Systems",
    "71": "Materials Science and Engineering (SS)",
    "72": "Material Science and Technology",
    "73": "Electronics and Communication Engineering",
    "74": "Pulp and Paper Technology",
    "75": "Tool Engineering",
    "76": "Artificial Intelligence and Data Science",
    "77": "Energy Engineering",
    "78": "Systems Engineering",
    "79": "Fashion Technology",
    "80": "Textile Technology (Fashion Technology)",
    "81": "Software Engineering",
    "82": "Environmental Engineering and Management",
    "83": "Smart Manufacturing",
    "84": "Mobile and Cloud Based Application (SS)",
    "85": "Bio-Technology (Food Technology)",
    "86": "Petroleum Engineering (SS)",
    "87": "Power System Engineering",
    "88": "Manufacturing Systems Management",
    "89": "Computer and Communication Engineering",
    "90": "Cloud Computing"}

    All_category = {"0":"OC","1":"BC","2":"MBC","3":"OC","4":"BCM","5":"BC","6":"SCA"}


    data = [x for x in request.form.values()]
    
    list1 = data.copy()

    #data.pop(4)
    data1 = [float(x) for x in data]

    final_output = np.array(data1).reshape(1, -1)
    output = model.predict(final_output)[0]

    list1.append(output[0])
    list1.append(output[1])
    list1.append(output[2])

    op0 = College_code[list1[3]]
    op1 = Branch_code[str(list1[4])]
    op2 = All_category[str(list1[5])]


    return render_template("index.html",prediction_text = "College: {}, Branch: {} , Allotted_Category: {}".format(op0,op1, op2), prediction = "Thank you, Hope this will match your requirement !!!")

if __name__ == '__main__':
    app.run(debug = True)
