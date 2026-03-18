import pandas as pd
import random

# -------------------------------
# Crop environmental requirements
# -------------------------------
crop_requirements = {

"Potato":{"temp":(15,20),"rain":(1200,2000)},
"Cabbage":{"temp":(15,25),"rain":(1000,1800)},
"Carrot":{"temp":(15,20),"rain":(800,1500)},
"Peas":{"temp":(12,18),"rain":(800,1200)},
"Beans":{"temp":(15,21),"rain":(900,1500)},
"Beetroot":{"temp":(15,20),"rain":(700,1200)},
"Radish":{"temp":(10,18),"rain":(700,1200)},
"Cauliflower":{"temp":(15,22),"rain":(900,1500)},
"KnolKhol":{"temp":(15,20),"rain":(800,1300)},
"Turnip":{"temp":(10,15),"rain":(700,1200)}
}

# -------------------------------
# Soil requirements (REALISTIC)
# -------------------------------
crop_soil_requirements = {

"Potato":{"ph":(5.0,6.5),"nitrogen":(250,400),"phosphorus":(40,80),"potassium":(150,300)},
"Cabbage":{"ph":(6.0,7.5),"nitrogen":(300,500),"phosphorus":(50,90),"potassium":(200,350)},
"Carrot":{"ph":(6.0,7.0),"nitrogen":(200,350),"phosphorus":(30,70),"potassium":(150,250)},
"Peas":{"ph":(6.0,7.5),"nitrogen":(150,300),"phosphorus":(40,80),"potassium":(100,200)},
"Beans":{"ph":(5.5,6.5),"nitrogen":(200,350),"phosphorus":(30,60),"potassium":(150,250)},
"Beetroot":{"ph":(6.0,7.0),"nitrogen":(200,350),"phosphorus":(30,70),"potassium":(150,250)},
"Radish":{"ph":(5.5,6.8),"nitrogen":(150,300),"phosphorus":(30,60),"potassium":(120,200)},
"Cauliflower":{"ph":(5.5,6.6),"nitrogen":(300,500),"phosphorus":(50,90),"potassium":(200,350)},
"KnolKhol":{"ph":(5.5,6.8),"nitrogen":(250,400),"phosphorus":(40,80),"potassium":(150,300)},
"Turnip":{"ph":(6.0,7.0),"nitrogen":(150,300),"phosphorus":(30,60),"potassium":(120,200)}
}

# -------------------------------
# Generate good soil data
# -------------------------------
def generate_soil(crop):
    values = crop_soil_requirements[crop]

    return {
        "soil_ph": round(random.uniform(*values["ph"]), 2),
        "nitrogen": round(random.uniform(*values["nitrogen"]), 2),
        "phosphorus": round(random.uniform(*values["phosphorus"]), 2),
        "potassium": round(random.uniform(*values["potassium"]), 2)
    }

# -------------------------------
# Generate bad soil (for realism)
# -------------------------------
def generate_bad_soil():
    return {
        "soil_ph": round(random.uniform(7.5, 9.0), 2),
        "nitrogen": round(random.uniform(50, 150), 2),
        "phosphorus": round(random.uniform(5, 20), 2),
        "potassium": round(random.uniform(50, 100), 2)
    }

# -------------------------------
# MAIN DATA GENERATION
# -------------------------------
rows = []

for crop, values in crop_requirements.items():

    for i in range(200):

        temperature = random.uniform(*values["temp"])
        rainfall = random.uniform(*values["rain"])

        # 🔥 ADD REALISTIC NOISE
        temperature += random.uniform(-2, 2)
        rainfall += random.uniform(-100, 100)

        soil = generate_soil(crop)

        humidity = min(95, max(40, rainfall / 25 + random.uniform(-10, 10)))
        irrigation = max(1, int(12 - rainfall / 200 + random.uniform(-2, 2)))

        # 🔥 BORDERLINE CASES (KEY FIX)
        if random.random() < 0.2:
            label = random.choice([0, 1])   # confusion zone
        else:
            label = 1

        rows.append({
            "crop": crop,
            "temperature": round(temperature, 2),
            "rainfall": round(rainfall, 2),
            "humidity": round(humidity, 2),
            "soil_ph": soil["soil_ph"],
            "nitrogen": soil["nitrogen"],
            "phosphorus": soil["phosphorus"],
            "potassium": soil["potassium"],
            "irrigation_days": irrigation,
            "label": label
        })

# -------------------------------
# ADD UNSUITABLE DATA (IMPORTANT)
# -------------------------------
for i in range(int(len(rows) * 0.2)):

    bad_soil = generate_bad_soil()

    rows.append({
        "crop": random.choice(list(crop_requirements.keys())),  # not "Unknown"
        "temperature": round(random.uniform(5, 35), 2),
        "rainfall": round(random.uniform(200, 2500), 2),
        "humidity": round(random.uniform(30, 95), 2),
        "soil_ph": bad_soil["soil_ph"],
        "nitrogen": bad_soil["nitrogen"],
        "phosphorus": bad_soil["phosphorus"],
        "potassium": bad_soil["potassium"],
        "irrigation_days": random.randint(1, 15),
        "label": 0
    })

# -------------------------------
# CREATE DATAFRAME
# -------------------------------
df = pd.DataFrame(rows)

# Shuffle dataset (VERY IMPORTANT)
df = df.sample(frac=1).reset_index(drop=True)

# Save dataset
df.to_csv("data/crop_dataset_realistic.csv", index=False)

print("✅ Dataset created successfully!")
print("Total samples:", len(df))
print(df.head())