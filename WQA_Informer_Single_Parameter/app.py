from flask import Flask, render_template, request
import pred_app  

app = Flask(__name__)

# Location mapping dictionary (ID -> Name)
location_mapping = {
    0: "Ganga US Lakhsmanjhula Rishikesh",
    1: "Ganga DS Swargashram-1 Rishikesh",
    2: "Ganga DS Bairaaj Rishikesh",
    3: "Ganga DS Lakkarghat Dehradun",
    4: "Ganga DS Raiwala Dehradun",
    5: "Ganga DS Birla Guest House Dehradun",
    6: "Ganga canal Rishikul Bridge Haridwar",
    7: "Ganga Har Ki Pauri Haridwar",
    8: "Ganga DS Bisanpur Kundi Haridwar",
    9: "Ganga canal LalTaRao Bridge Haridwar",
    10: "Ganga canal Damkothi Haridwar",
    11: "Ganga US Bindughat Dudhiyaban Haridwar",
    12: "Ganga DS Balkumari Mandir Ajeetpur Haridwar",
    13: "Ganga canal DS Roorkee Haridwar",
    14: "Ganga DS Sultanpur Haridwar"
}

# Convert dict to list of dicts for template table
locations_list = [{"Location ID": k, "Location Name": v} for k, v in location_mapping.items()]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    error = None
    selected_location_name = None
    selected_location_id = None
    target_month = None
    target_year = None

    if request.method == "POST":
        try:
            selected_location_id = int(request.form.get("location_num"))
            target_month = int(request.form.get("target_month"))
            target_year = int(request.form.get("target_year"))

            result = pred_app.predict_bod(selected_location_id, target_year, target_month)

            if result is None:
                error = "❌ Not enough data to create sequence for this location."

            selected_location_name = location_mapping.get(selected_location_id, "Unknown Location")

        except Exception as e:
            error = f"Error: {str(e)}"
            result = None

    return render_template(
        "predict.html",
        location_df=locations_list,
        result=result,
        error=error,
        selected_location_name=selected_location_name,
        selected_location_id=selected_location_id,
        target_month=target_month,
        target_year=target_year
    )

if __name__ == "__main__":
    app.run(debug=True)
