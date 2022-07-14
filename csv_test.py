# import csv

# for i in range(10):
#     # rows = [
#     #     ["Mando", "Mandalorian", "Bounty Hunter", 35],
#     #     ["Grogu", "Mandalorian", "Jedi Master", 50],
#     #     ["Eleven", "Stranger Things", "Kid", 14],
#     #     ["Jon", "Game of Thrones", "King", 30],
#     #     ["Ross", "Friends", "Paleontologist", 35],
#     # ]

#     str_i = i
#     str_i = str(str_i)

#     with open("test.csv", "w") as f:

#         # using csv.writer method from CSV package
#         write = csv.writer(f)

#         write.writerows(str_i)
#     i += 1

text_file = open(r"test.txt", "w")
my_string = "type your string here"
text_file.write(my_string)
text_file.close()

text_file = open(r"test.txt", "w")
raw_data = ""
