import pandas as pd

dict_one = {"A":1, "B":2, "C":3}
dict_two = {"A":10, "B":20, "C":30}
plcaeholder = {"A":0, "B":0, "C":0}

test_dict = {"1":{}, "4":{}}
test_dict["1"] = plcaeholder.copy()
test_dict["4"] = plcaeholder.copy()

test_dict["1"]["A"] = 1
test_dict["4"]["B"] = 5
# print("plcaeholder: ")
# print(plcaeholder)

# print("test_dict: ")
# print(test_dict)

dataframe = pd.DataFrame.from_dict(test_dict)
# dataframe = pd.DataFrame.from_dict(dict_one, orient='index', columns=['One'] )
# dataframeTwo =  pd.DataFrame.from_dict(dict_two, orient='index', columns=['Two'] )
# dataframe.insert(0, "two", dict_two.values())
# print(dataframe)
print(dataframe)

print("Done!")