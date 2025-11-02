
import importlib
import json
import mip
import cffi
importlib.reload(cffi)

with open("/content/test02.json", 'r') as f:
    data = json.load(f)

# Extract data
days = data["days"]
rooms = data["rooms"]
patients = data["patients"]
nurses = data["nurses"]

# Define sets
P = range(len(patients))
R = range(len(rooms))
D = range(days)
N = range(len(nurses))

# Define parameters
release_date = {p["id"]: p["release_date"] for p in patients}
due_date = {p["id"]: p["due_date"] for p in patients}
length_of_stay = {p["id"]: p["length_of_stay"] for p in patients}
incompatible_rooms = {p["id"]: p["incompatible_room_ids"] for p in patients}
capacity = {r["id"]: r["capacity"] for r in rooms}
working_shifts = {n["id"]: {shift["day"] for shift in n["working_shifts"]} for n in nurses}


# Model
model = mip.Model()

# Decision variables
x = model.add_var_tensor((len(P), len(R), len(D)), var_type=mip.BINARY, name="x")  # x(prd) : 1 if patient p is assigned to room r on day d
z = model.add_var_tensor((len(P), len(D)), var_type=mip.BINARY, name="z")         # z(pd) : 1 if patient p is admitted on day d
y = model.add_var_tensor((len(R), len(N), len(D)), var_type=mip.BINARY, name="y") # y(rnd): 1 if nurse n is assigned to room r on day d

# Objective: Minimize total admission delay
model.objective = mip.minimize(
    mip.xsum(
        z[p, d] * (d - release_date[p])
        for p in P for d in D
    )
)

# Constraints

# 1. Admission days must be within the specified release and due dates
for p in P:
    for d in D:
        if d < release_date[p] or d > due_date[p] - length_of_stay[p] + 1:  # +1 to include the last possible admission day
            model += z[p, d] == 0


# 2. Admission must respect release and due dates
for p in P:
    for d in D:
        if d >= release_date[p] and d <= due_date[p] - length_of_stay[p] + 1:
            for r in R:
              model += mip.xsum(x[p, r, d + i] for i in range(min(length_of_stay[p], days - d))) == z[p, d] * length_of_stay[p]
            else:  # No admission possible in this timeframe
               model += z[p, d] == 0


# 3. Rooms cannot exceed capacity on any day
for r in R:
    for d in D:
        model += mip.xsum(x[p, r, d] for p in P) <= capacity[r]

# 4. Patients cannot be assigned to incompatible rooms
for p in P:
    for r in R:  # Iterate through all rooms
        if rooms[r]["id"] in incompatible_rooms[p]:  # Check if room is incompatible for patient
            for d in D:
                model += x[p, r, d] == 0

# 5. Each occupied room has exactly one nurse per day
for r in R:
    for d in D:
        model += mip.xsum(y[r, n, d] for n in N) == mip.xsum(x[p, r, d] for p in P)

# 6. Nurses can only be assigned on their working days
for n in N:
    for d in D:
        if d not in working_shifts[n]:  # Assuming days in working_shifts are 0-indexed
            for r in R:
                model += y[r, n, d] == 0
# 7. Nurses can be assigned to a maximum of 3 rooms in the same day
for n in N:
    for d in D:
        model += mip.xsum(y[r, n, d] for r in R) <= 3



# Solve the model
status = model.optimize()

print(f"Optimal total admission delay = {model.objective_value}")