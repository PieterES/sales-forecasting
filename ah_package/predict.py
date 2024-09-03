import os
import sys

"""
    Create an easier way for the user to import. Instead of having to input the entire curl string, the user can call this script with only the port number and the list of inputs as a json list. 
"""
if len(sys.argv) != 3:
    print("Incorrect inputs. Either port number or the list of inputs is missing.")
    print("Correct Usage Example:")
    print("python ah_package/predict.py 8000 '[781, 12602.0, 4922.0, true, 8646, 7292, 5494, 11, 3, 6.190, 6.217, 6.075]'")
    sys.exit(1)

port = sys.argv[1]
json_list = sys.argv[2]

os.system(
    f'curl -X POST http://localhost:{port}/predict -H "Content-Type: application/json" -d "{json_list}"'
)
