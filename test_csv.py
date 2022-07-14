import csv

cols = ["Character Name", "Series Name", "Profession", "Age"]

rows = [
    ["Mando", "Mandalorian", "Bounty Hunter", 35],
    ["Grogu", "Mandalorian", "Jedi Master", 50],
    ["Eleven", "Stranger Things", "Kid", 14],
    ["Jon", "Game of Thrones", "King", 30],
    ["Ross", "Friends", "Paleontologist", 35],
]

with open("shows.csv", "w") as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(cols)
    write.writerows(rows)
