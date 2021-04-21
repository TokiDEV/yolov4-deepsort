import sys
import json


def read_in():
    line = sys.stdin.readline()
    # Since our input would only be having one line, parse our JSON data from that
    return json.loads(line)


def main():
    while True:
        data = read_in()
        image = data['frame_image']
        shape = data['frame_shape']
        id = data['frame_id']
        print(json.dumps({
            'frame_id': id,
            'result': {
                'fps': 0,
                'tracks': []
            }
        }, indent=4))

if __name__ == '__main__':
    main()
