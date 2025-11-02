# Patient-Nurse-Room-Assignment

In this project, We Need to solve a healthcare scheduling problem involving the assignment of patients to rooms and nurses to rooms in a hospital. The goal is to develop an optimization model using mip that minimizes the total delay in patient admissions while adhering to a set of operational and resource constraints. This problem is inspired by the Integrated Healthcare Timetabling Competition 2024 (https://ihtc2024.github.io/).

The hospital operates with a set of patients, rooms, and nurses, each with specific characteristics. Patients are defined by their release date, indicating the earliest day they can be admitted, and a due date, specifying the latest allowable admission day. Patients also have a required length of stay. Rooms are characterized by fixed capacities that determine the maximum number of patients they can accommodate at any given time. Additionally, some patients may have incompatibilities with certain rooms, restricting their assignment.

Nurses have predefined rosters detailing their availability across days. Each occupied room must have one assigned nurse on every day of the scheduling horizon. Nurses can be assigned to a maximum of 3 rooms in a given day.

The problem requires assigning each patient to a room and ensuring that every occupied room has an assigned nurse during each day. These assignments must satisfy several constraints: no room may exceed its capacity, nurse assignments must respect their availability, and patients must only be assigned to compatible rooms. The primary objective is to minimize the total delay in admissions, defined as the sum of the differences between actual admission dates and the release dates of all patients. In what follows we detail the constraints and decisions.

Constraints:

A patient’s length of stay must be respected, meaning they occupy a room for the full duration of their stay.
A room cannot exceed its capacity on any given day.
Patients can only be assigned to rooms that they are compatible with.
Each occupied room must have exactly one assigned nurse per day.
Nurses can only be assigned to rooms on days when they are available.
Nurses can be assigned to a maximum of 3 different rooms in the same day.
Admission days must be within the specified release and due dates.
Decisions:

Assign each patient to a room, ensuring that the same room is assigned for the duration of their stay.
Determine the admission day for each patient within the allowed range of their release and due dates.
Assign one nurse to each occupied room on every day of the scheduling period.
The objective is to minimize the total admission delay, defined as the sum of the differences between the admission day and the release date of patients.

