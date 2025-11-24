import pickle
import numpy as np

with open("er_load_model.pkl", "rb") as f:
    model = pickle.load(f)

def get_user_input():
    print("\nEnter the following details:\n")

    temperature = float(input("Temperature: "))
    aqi = float(input("AQI (Air Quality Index): "))
    festival = int(input("Festival today? (0 = No, 1 = Yes): "))
    current_patients = int(input("Current ER patients: "))
    humidity = float(input("Humidity (%): "))
    day_type = int(input("Day Type (0 = Weekday, 1 = Weekend): "))

    return np.array([[temperature, aqi, festival, current_patients, humidity, day_type]])


def er_load_details(category):
    mapping = {
        0: ("Low", 5),
        1: ("Moderate", 15),
        2: ("High", 35),
        3: ("Critical", 55)
    }
    return mapping.get(category, ("Unknown", 0))


def staff_recommendation(category):
    if category == 0:
        return 2, 3, "Normal load. Regular staffing.", "Normal"
    if category == 1:
        return 3, 5, "Moderate load. Add 1 extra nurse.", "Moderate"
    if category == 2:
        return 5, 9, "High load. Add 1 doctor + 2 nurses.", "High"
    if category == 3:
        return 7, 12, "Critical load. Activate emergency staff.", "Critical"


def resource_preparation(category):
    oxygen_map = {
        0: ("Low", 8),
        1: ("Medium", 10),
        2: ("Medium", 12),
        3: ("High", 18)
    }

    icu_map = {
        0: ("Low", 1),
        1: ("Medium", 2),
        2: ("High", 4),
        3: ("Critical", 6)
    }

    ppe_map = {
        0: ("Low", 15),
        1: ("Medium", 20),
        2: ("High", 25),
        3: ("Critical", 35)
    }

    return oxygen_map[category], icu_map[category], ppe_map[category]


user_data = get_user_input()
prediction = int(model.predict(user_data)[0])

er_label, expected_patients = er_load_details(prediction)
staff_doctors, staff_nurses, staff_msg, staff_label = staff_recommendation(prediction)
oxygen, icu, ppe = resource_preparation(prediction)

print("\n==============================")
print("       ER LOAD PREDICTION")
print("==============================")
print(f"Load Category: {er_label} ({prediction})")
print(f"Expected Patients (next 6 hrs): ~{expected_patients}")

print("\n==============================")
print("     STAFF RECOMMENDATION")
print("==============================")
print(f"Required Doctors: {staff_doctors}")
print(f"Required Nurses: {staff_nurses}")
print(f"Alert: {staff_msg}")
print(f"Staffing Level: {staff_label}")

print("\n==============================")
print("   RESOURCE PREPARATION")
print("==============================")

oxy_label, oxy_qty = oxygen
print(f"Oxygen Level: {oxy_label}")
print(f"Cylinders Needed: {oxy_qty}")

icu_label, icu_qty = icu
print(f"\nICU Bed Risk: {icu_label}")
print(f"Beds to Prepare: {icu_qty}")

ppe_label, ppe_qty = ppe
print(f"\nPPE Requirement: {ppe_label}")
print(f"Kits Needed: {ppe_qty}")

print("\n==============================\n")
