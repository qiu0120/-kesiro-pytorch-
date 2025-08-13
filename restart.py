import directkeys
import time

def restart():
    print("死,restart")
    time.sleep(8)
    directkeys.attack()
    time.sleep(0.2)
    directkeys.lock_vision()
    print("开始新一轮")
    
if __name__ == '__main__':
    time.sleep(15)
    restart()