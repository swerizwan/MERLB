file = 'users.dat'
def add(channel: str):
    f = open(file, 'a')
    f.write('\n')
    f.write(channel)
    f.close()
    
def remove(channel: str):
    if (channel != "deadfracture"):
        f = open(file,'r')
        lst = []
        for line in f:
            line = line.replace(channel,'')
            if (line != '\n'):
                lst.append(line)
        f.close()
        f = open(file,'w')
        for line in lst:
            f.write(line)
        f.close()
        
def list():
    with open('users.dat') as f:
        lines = [line.rstrip() for line in f]
        return(lines)
        
