import pickle as cPickle

common_dict = cPickle.load( open( "common_dict.save", "rb" ) )

for k in sorted(common_dict.keys()):
	
	print(k)

print(len(common_dict))
print(type(common_dict))
