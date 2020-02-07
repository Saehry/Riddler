partial_patts = ['(']
full_patts = []
n = 5
while partial_patts:
	p = partial_patts.pop()
	if len(p) == 2*n:
		full_patts.append(p)
	if p.count('(') < n:
		partial_patts.append(p + '(')
	if p.count('(') > p.count(')'):
		partial_patts.append(p + ')')

print(full_patts, len(full_patts))

vals = set()
for p in full_patts:
	p_string = "".join([x for t in zip(list(p), map(lambda y: str(y), range(-1, -2*n, -1))) for x in t] + [')'])
	p_string = p_string.replace('(', '*abs(')[1:]
	#p_string = p_string.replace(')', ')*')[:-1]
	print(p_string, eval(p_string))
	vals.add(eval(p_string))

vals = list(vals)
vals.sort()

print(vals, len(vals))

	
