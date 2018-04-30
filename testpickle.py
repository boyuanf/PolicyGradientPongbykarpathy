import pickle

testp = pickle.load(open("save_from_c5.p", "rb"))
print(testp)


a = {'hello': 'world'}

with open('filename.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)

print (a == b)


favorite_color = {"lion": "yellow", "kitty": "red"}  # create a dictionary
pickle.dump(favorite_color, open("test.p", "wb"))  # save it into a file named save.p
#favorite_color = pickle.load(open("test.p", "rb"), encoding='iso-8859-1')
favorite_color = pickle.load(open("test.p", "rb"))
print(favorite_color)

example_dict = {1:"6",2:"2",3:"f"}

pickle_out = open("dict.pickle","wb")
pickle.dump(example_dict, pickle_out)
pickle_out.close()

pickle_in = open("dict.pickle","rb")
example_dict = pickle.load(pickle_in)

print(example_dict)

