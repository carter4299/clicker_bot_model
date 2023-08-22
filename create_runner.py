from os import environ
from time import sleep
from sys import exit

from PIL import ImageGrab
import numpy as np
from pyautogui import click, moveTo
from keras.preprocessing import image
from tensorflow.keras.models import load_model

from logger.log import logger as log

HOME_PAGE = None
def store_home_page(img):
    global HOME_PAGE
    HOME_PAGE = img
def get_home_page():
    global HOME_PAGE
    return HOME_PAGE

def recover(max_tries=10):
    log.info("Trying to recover, before exiting ...")
    home = get_home_page()
    t=0
    while t < max_tries:
        img = high_res()
        #moveTo( INSERT YOUR BACK OR ESCAPE BUTTON LOCATION HERE )
        click()
        if detect_change(np.array(low_res(home)), np.array(low_res(img))):
            log.info("Recovery success, restarting script ...")
            return True
        else:
            log.info("Page not changed, waiting ...")
            sleep(3)
        t+=1
    
    return False

def safe_exit():
    log.info("Trying to recover, before exiting ...")
    if recover():
        main()
    else:
        log.info("Recovery failed, exiting ...")
        exit(0)

def high_res():
    return ImageGrab.grab()
    
def low_res(img, downsample_factor=0.5):
    return img.resize((int(img.size[0] * downsample_factor), int(img.size[1] * downsample_factor))).resize((img.size[0], img.size[1]))

def detect_change(prev_img, curr_img, threshold=5):
    return np.mean(np.abs(prev_img - curr_img)) > threshold

def wait_for_page(interval=1, tries=0, ret_image=False):
    img = high_res()
    sleep(interval)
    _img = high_res()
    if detect_change(np.array(low_res(img)), np.array(low_res(_img))):
        log.info("Page changed, moving on ...")
        if ret_image:
            return _img
        return
    elif tries>20 and interval>=1:
        log.info("Bug detected, restarting ...")
        safe_exit()
    elif tries>50 and interval<1:
        log.info("Bug detected, restarting ...")
        safe_exit()
    else:
        log.info("Page not changed, waiting ...")
        wait_for_page(interval=interval,tries=tries+1)

class CreateRunner:
    def __init__(self, model):
        self.large_mode_interval = BOT
        self.small_mode_interval = 0.1 * BOT
        self.model = model
        self.img_width, self.img_height = 28, 28
        self.common_click_location1 = [0,0]
        self.common_click_location2 = [0,0]
        self.common_click_location3 = [0,0]
        self.common_click_location4 = [0,0]
        self.max_price, # first 2 ints of 5 digit num
        self.padding, #padding before first number
        self.num_width, self.comma_width, #width of number and comma/decimal
        self.num_of_num_blocks, #number of number blocks needed
        self.starting_y_block_crop, #starting location of the first number block
        self.increment_y_block_crop, #increment of the y location of the number block
        self.starting_x_block_crop, #starting location of the first number block
        self.increment_x_block_crop, #increment of the x location of the number block

        #for multiple number blocks, if blocks align vertically, use y, if horizontally, use x
        #if you need more than 1 regions, add more to the list    ------------------------------------------\
        self.all_x_crop = [self.starting_x_block_crop + (self.increment_x_block_crop * i) for i in range(0, self.num_of_num_blocks)] 
        # or 
        self.all_y_crop = [self.starting_y_block_crop + (self.increment_y_block_crop * i) for i in range(0, self.num_of_num_blocks)]

        #example of a missing digit identifier, the program I used have yellow pixels if the digit was of length 4 instead of 5
        self.missing_digit_identifier = {(0.5764706, 0.5647059, 0.23137255),(0.6117647, 0.56078434, 0.21176471),(0.5686275, 0.5529412, 0.3647059),(0.60784316, 0.61960787, 0.38431373),(0.69803923, 0.6117647, 0.34901962),(0.627451, 0.627451, 0.39607844),(0.88235295, 0.85490197, 0.6156863),(0.7254902, 0.6745098, 0.42745098),(0.67058825, 0.61960787, 0.27058825),(0.56078434, 0.56078434, 0.28627452),(0.9098039, 0.84313726, 0.5803922),(0.9098039, 0.85882354, 0.61960787),(0.49803922, 0.49803922, 0.37254903),(0.8980392, 0.84705883, 0.60784316),(0.6039216, 0.5882353, 0.2901961),(0.6509804, 0.5803922, 0.23921569),(0.7019608, 0.65882355, 0.3647059),(0.6313726, 0.6392157, 0.43137255),(0.6313726, 0.5921569, 0.3137255),(0.6627451, 0.6117647, 0.3647059),(0.7372549, 0.6745098, 0.34117648),(0.7372549, 0.6666667, 0.3254902),(0.92156863, 0.8745098, 0.7294118),(0.69803923, 0.6392157, 0.28627452),(0.70980394, 0.7058824, 0.49411765),(0.7254902, 0.6392157, 0.3764706),(0.84313726, 0.78431374, 0.43137255),(0.7647059, 0.74509805, 0.5764706),(0.6313726, 0.5803922, 0.23137255),(0.6117647, 0.627451, 0.41568628),(0.7019608, 0.69803923, 0.4862745),(0.8666667, 0.8, 0.6),(0.6392157, 0.6, 0.32156864),(0.654902, 0.6156863, 0.3372549),(0.6901961, 0.6313726, 0.2784314),(0.8039216, 0.7411765, 0.40784314),(0.8627451, 0.827451, 0.60784316),(0.9137255, 0.84705883, 0.58431375),(0.6392157, 0.5882353, 0.23921569),(0.65882355, 0.60784316, 0.36078432),(0.8156863, 0.8039216, 0.64705884),(0.6117647, 0.6156863, 0.42745098),(0.65882355, 0.61960787, 0.34117648),(0.654902, 0.59607846, 0.24313726),(0.7490196, 0.7058824, 0.4117647),(0.60784316, 0.62352943, 0.4117647),(0.6627451, 0.61960787, 0.3254902),(0.89411765, 0.8509804, 0.60784316),(0.69411767, 0.69411767, 0.41960785),(0.6745098, 0.654902, 0.4862745),(0.9098039, 0.84313726, 0.6431373),(0.7019608, 0.6431373, 0.2901961),(0.74509805, 0.65882355, 0.39607844),(0.87058824, 0.8235294, 0.6784314),(0.7411765, 0.69803923, 0.40392157),(0.6862745, 0.69411767, 0.4862745),(0.6039216, 0.6156863, 0.38039216),(0.87058824, 0.8039216, 0.5411765),(0.7372549, 0.6784314, 0.3254902),(0.62352943, 0.62352943, 0.39215687),(0.63529414, 0.61960787, 0.32156864),(0.8, 0.7490196, 0.5019608),(0.7058824, 0.63529414, 0.29411766),(0.64705884, 0.64705884, 0.37254903),(0.6313726, 0.6313726, 0.35686275),(0.90588236, 0.8392157, 0.6392157),(0.8509804, 0.8235294, 0.58431375),(0.90588236, 0.8627451, 0.61960787),(0.88235295, 0.84705883, 0.627451),(0.5647059, 0.5529412, 0.21960784),(0.5647059, 0.54901963, 0.2509804),(0.84313726, 0.7921569, 0.5529412),(0.6313726, 0.63529414, 0.44705883),(0.6313726, 0.6156863, 0.42745098),(0.6431373, 0.57254905, 0.23137255),(0.69803923, 0.63529414, 0.3019608),(0.9411765, 0.9137255, 0.6745098),(0.7647059, 0.7529412, 0.59607846),(0.7294118, 0.67058825, 0.31764707),(0.84313726, 0.78039217, 0.44705883),(0.8039216, 0.74509805, 0.39215687),(0.8666667, 0.8, 0.5372549),(0.67058825, 0.58431375, 0.32156864),(0.6862745, 0.6784314, 0.5647059),(0.5568628, 0.54509807, 0.21176471),(0.6313726, 0.6156863, 0.31764707)}


    def perform_action(self, action):
        #enter your action here
        """
        moveTo((x, y)
        click()
        wait_for_page(interval=z)
        """
        pass
    
    def buy_signal(self):
        """moveTo(x, y)
        click()
        img = wait_for_page(interval=self.small_mode_interval, ret_image=True)
        self.extract_numbers(img)
        log.error("Bug detected, restarting ...")
        safe_exit()"""
        pass

    def crop_img_to_get_numbers(self, frame): # Returns a single cropped image of the number block, use loop to get all numbers
        return frame.crop((self.starting_x_block_crop, self.starting_y_block_crop, self.increment_x_block_crop+self.starting_x_block_crop, self.increment_y_block_crop+self.starting_y_block_crop))
    
    def convert_to_grayscale(self, img):
        return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    
    def get_pixels(self, img):
        return [tuple(p[:3]) for p in img.reshape(-1, img.shape[2])]
    
    def euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    def almost_equal(self, pixels, missing_digit_identifier, threshold=0.05):
        for p in pixels:
            for g in missing_digit_identifier:
                if self.euclidean_distance(np.array(p), np.array(g)) < threshold:
                    return True
        return False
    
    def check_price(self, img):
        img_array = image.img_to_array(img.resize((28, 28)).convert('L'))
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = self.model.predict(img_array)
        predicted_number = np.argmax(predictions[0])
        
        return predicted_number

    def extract_numbers(self,frame): # the frame at this point is the screen shot of the page
        """
        block = self.crop_img_to_get_numbers(frame)
        padding_plus_num_width = self.padding + self.num_width
        # get the first 2 numbers of the block, if you need more adjust accordingly
        init = padding + num_width
        num1_img = first_block[:, init:init + self.num_width]
        init += num_width
        num2_img = first_block[:, init:init + self.num_width]
        if self.almost_equal(self.get_pixels(num1_img), self.missing_digit_identifier) or self.almost_equal(self.get_pixels(num2_img), self.missing_digit_identifier):
            # identifier to let you know the number is less than expected 
            { e.g. 
                expect 5 digits 100.00. 
                If the first num_img does not contain mostly the number color or a specific color, then we know the number is less than 100.00
            }
            self.perform_action()
        else:
            # get the number
            num1 = self.check_price(num1_img)
            num2 = self.check_price(num2_img)
            if num1 * 10 + num2 < self.max_price:
                self.perform_action()
          
        # number is greater than min limit
        # end of extract_numbers action
        """
        pass

    def runner(self):
        store_home_page(high_res())
        log.info("Starting ... ")
        self.main_page()
        log.error("Bug detected, restarting ...")
        safe_exit()
           
def main():
    environ['CUDA_VISIBLE_DEVICES'] = ''
    model = load_model('./trained/active.h5')

    al = CreateRunner(model)
    al.runner()

if __name__ == "__main__":
    """FASTEST"""
    BOT = 0.25
    """FAST"""
    BOT = 0.5    
    """STANDARD"""
    BOT = 1
    """RELAXED"""
    BOT = 1.5
    """NIGHT BOT"""
    BOT = 2