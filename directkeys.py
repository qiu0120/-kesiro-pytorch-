from pynput import keyboard
from pynput import mouse
from pynput.mouse import Button
import time

keybd = keyboard.Controller()
mouse = mouse.Controller()

def defense():
    mouse.press(Button.right)
    time.sleep(0.05)
    mouse.release(Button.right)
    time.sleep(0.1)
    
def attack():
    mouse.press(Button.left)
    time.sleep(0.05)
    mouse.release(Button.left)
    time.sleep(0.1)

def go_forward():
    keybd.press('w')
    time.sleep(0.3)
    keybd.release('w')

def go_back():
    keybd.press('s')
    time.sleep(0.1)
    keybd.release('s')

def go_left():
    keybd.press('a')
    time.sleep(0.1)
    keybd.release('a')

def go_right():
    keybd.press('d')
    time.sleep(0.1)
    keybd.release('d')

def jump():
    keybd.press(keyboard.Key.space)
    time.sleep(0.1)
    keybd.release(keyboard.Key.space)
    time.sleep(0.2)

def dodge():  # 闪避/识破
    keybd.press('w')
    keybd.press(keyboard.Key.shift)
    time.sleep(0.1)
    keybd.release(keyboard.Key.shift)
    keybd.release('w')

def lock_vision():
    mouse.press(Button.middle)
    time.sleep(0.1)
    mouse.release(Button.middle)
    time.sleep(0.1)
    
def use_tool():
    keybd.press('r')
    time.sleep(0.5)
    keybd.release('r')
    time.sleep(0.2)

if __name__ == '__main__':
    print(mouse.position)
    time.sleep(5)
    mouse.press(Button.left)
    # mouse.position = (1249,937)
    # attack()
    # time.sleep(1)
    # go_forward()
    # time.sleep(1)
    # go_back()
    # time.sleep(1)
    # go_left()
    # time.sleep(1)
    # go_right()
    # time.sleep(1)
    # jump()
    # time.sleep(1)
    # dodge()
    # time.sleep(1)
    # lock_vision()
    # time.sleep(1)
    # defense()
    # time.sleep(1)
    
    