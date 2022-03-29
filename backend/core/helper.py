def modify_input_for_multiple_files(image):
    d={}
    count=0
    for img in image:
        d[count]=img[str(count)]
        count+=1
    return d