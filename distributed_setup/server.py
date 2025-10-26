import pickle
import socket


def set_server_connection(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except Exception:
        pass
    server_socket.bind((host, port))
    server_socket.listen(5)
    print('The server has started!')
    return server_socket


def send_data(s, data):
    print('Sending data')
    data = pickle.dumps(data)
    s.sendall(data)
    return len(data)
