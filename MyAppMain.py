from myapp import app
app.run(host='0.0.0.0', port=3001, debug=True)
import os
import atexit

def exit_handler():
    for file in os.listdir('myapp/static'): 
        if (file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg')):
            os.remove(file)
            os.remove('myapp/static/'+file)

atexit.register(exit_handler)


