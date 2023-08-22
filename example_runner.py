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
        moveTo(236,1026)
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

class MainLoop:
    def __init__(self, model):
        self.large_mode_interval = BOT
        self.small_mode_interval = 0.1 * BOT
        self.model = model
        self.img_width, self.img_height = 28, 28
        self.home_page = None
        self.buy_box = [[473,572],[675,744]]
        self.back_button_x, self.back_button_y = 236,1026
        self.main_page_x, self.main_page_y = 786,424
        self.quality_x, self.quality_y = 253, 424
        self.quality_choice_x, self.quality_choice_y = 970, 527
        self.max_price = 30 # first 2 ints of 5 digit num
        self.padding = -8
        self.num_width, self.comma_width = 10, 5
        self.en_x = self.padding + self.num_width
        self.y = 667
        self._y = 25
        self.y_ = self.y+self._y
        self.x = 612
        self._x = 59
        self.x_ = 229
        self._x_ = [self.x+(self.x_*i) for i in range(0, 6)]
        self.__x_ = [self.x+self._x for x in self._x_]
        self.crop_ = {"region1": [(self._x_[0], self.y), (self.__x_[0], self.y_)],"region2": [(self._x_[1], self.y), (self.__x_[1], self.y_)],"region3": [(self._x_[2], self.y), (self.__x_[2], self.y_)],"region4": [(self._x_[3], self.y), (self.__x_[3], self.y_)],"region5": [(self._x_[4], self.y), (self.__x_[4], self.y_)],"region6": [(self._x_[5], self.y), (self.__x_[5], self.y_)],}
        self.missing_digit_id = {(0.5764706, 0.5647059, 0.23137255),(0.6117647, 0.56078434, 0.21176471),(0.5686275, 0.5529412, 0.3647059),(0.60784316, 0.61960787, 0.38431373),(0.69803923, 0.6117647, 0.34901962),(0.627451, 0.627451, 0.39607844),(0.88235295, 0.85490197, 0.6156863),(0.7254902, 0.6745098, 0.42745098),(0.67058825, 0.61960787, 0.27058825),(0.56078434, 0.56078434, 0.28627452),(0.9098039, 0.84313726, 0.5803922),(0.9098039, 0.85882354, 0.61960787),(0.49803922, 0.49803922, 0.37254903),(0.8980392, 0.84705883, 0.60784316),(0.6039216, 0.5882353, 0.2901961),(0.6509804, 0.5803922, 0.23921569),(0.7019608, 0.65882355, 0.3647059),(0.6313726, 0.6392157, 0.43137255),(0.6313726, 0.5921569, 0.3137255),(0.6627451, 0.6117647, 0.3647059),(0.7372549, 0.6745098, 0.34117648),(0.7372549, 0.6666667, 0.3254902),(0.92156863, 0.8745098, 0.7294118),(0.69803923, 0.6392157, 0.28627452),(0.70980394, 0.7058824, 0.49411765),(0.7254902, 0.6392157, 0.3764706),(0.84313726, 0.78431374, 0.43137255),(0.7647059, 0.74509805, 0.5764706),(0.6313726, 0.5803922, 0.23137255),(0.6117647, 0.627451, 0.41568628),(0.7019608, 0.69803923, 0.4862745),(0.8666667, 0.8, 0.6),(0.6392157, 0.6, 0.32156864),(0.654902, 0.6156863, 0.3372549),(0.6901961, 0.6313726, 0.2784314),(0.8039216, 0.7411765, 0.40784314),(0.8627451, 0.827451, 0.60784316),(0.9137255, 0.84705883, 0.58431375),(0.6392157, 0.5882353, 0.23921569),(0.65882355, 0.60784316, 0.36078432),(0.8156863, 0.8039216, 0.64705884),(0.6117647, 0.6156863, 0.42745098),(0.65882355, 0.61960787, 0.34117648),(0.654902, 0.59607846, 0.24313726),(0.7490196, 0.7058824, 0.4117647),(0.60784316, 0.62352943, 0.4117647),(0.6627451, 0.61960787, 0.3254902),(0.89411765, 0.8509804, 0.60784316),(0.69411767, 0.69411767, 0.41960785),(0.6745098, 0.654902, 0.4862745),(0.9098039, 0.84313726, 0.6431373),(0.7019608, 0.6431373, 0.2901961),(0.74509805, 0.65882355, 0.39607844),(0.87058824, 0.8235294, 0.6784314),(0.7411765, 0.69803923, 0.40392157),(0.6862745, 0.69411767, 0.4862745),(0.6039216, 0.6156863, 0.38039216),(0.87058824, 0.8039216, 0.5411765),(0.7372549, 0.6784314, 0.3254902),(0.62352943, 0.62352943, 0.39215687),(0.63529414, 0.61960787, 0.32156864),(0.8, 0.7490196, 0.5019608),(0.7058824, 0.63529414, 0.29411766),(0.64705884, 0.64705884, 0.37254903),(0.6313726, 0.6313726, 0.35686275),(0.90588236, 0.8392157, 0.6392157),(0.8509804, 0.8235294, 0.58431375),(0.90588236, 0.8627451, 0.61960787),(0.88235295, 0.84705883, 0.627451),(0.5647059, 0.5529412, 0.21960784),(0.5647059, 0.54901963, 0.2509804),(0.84313726, 0.7921569, 0.5529412),(0.6313726, 0.63529414, 0.44705883),(0.6313726, 0.6156863, 0.42745098),(0.6431373, 0.57254905, 0.23137255),(0.69803923, 0.63529414, 0.3019608),(0.9411765, 0.9137255, 0.6745098),(0.7647059, 0.7529412, 0.59607846),(0.7294118, 0.67058825, 0.31764707),(0.84313726, 0.78039217, 0.44705883),(0.8039216, 0.74509805, 0.39215687),(0.8666667, 0.8, 0.5372549),(0.67058825, 0.58431375, 0.32156864),(0.6862745, 0.6784314, 0.5647059),(0.5568628, 0.54509807, 0.21176471),(0.6313726, 0.6156863, 0.31764707)}

    def buy_item(self):
        moveTo((self.buy_box[0][0]+self.buy_box[1][0])//2, (self.buy_box[0][1]+self.buy_box[1][1])//2)
        click()
        """
            x,y = coordinates of button
        """ 
        wait_for_page(interval=self.small_mode_interval)
        #moveTo(x, y)
        #click()
        wait_for_page(interval=self.small_mode_interval)
        #moveTo(x, y)
        #click()
        return
    
    def back_button(self):
        moveTo(self.back_button_x, self.back_button_y)
        click()
        wait_for_page()
        self.main_page()
        log.error("Bug detected, restarting ...")
        safe_exit()
    
    def quality_choice(self):
        moveTo(self.quality_choice_x, self.quality_choice_y)
        click()
        img = wait_for_page(interval=self.small_mode_interval, ret_image=True)
        self.extract_numbers(img)
        log.error("Bug detected, restarting ...")
        safe_exit()
    
    def quality(self):
        moveTo(self.quality_x, self.quality_y)
        click()
        wait_for_page(interval=self.small_mode_interval)
        self.quality_choice()
        log.error("Bug detected, restarting ...")
        safe_exit()
    
    def main_page(self):
        moveTo(self.main_page_x, self.main_page_y)
        click()
        wait_for_page(interval=self.small_mode_interval)
        self.quality()
        log.error("Bug detected, restarting ...")
        safe_exit()

    def price_block(self, frame):
        return frame.crop((self._x_[0], self.y, self.__x_[0], self.y_))
    
    def convert_to_grayscale(self, img):
        return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    
    def get_pixels(self, img):
        return [tuple(p[:3]) for p in img.reshape(-1, img.shape[2])]
    
    def euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    def almost_equal(self, pixels, missing_digit_id, threshold=0.05):
        for p in pixels:
            for g in missing_digit_id:
                if self.euclidean_distance(np.array(p), np.array(g)) < threshold:
                    return True
        return False
    
    def check_price(self, img):
        img_array = image.img_to_array(img.resize((28, 28)).convert('L'))
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = self.model.predict(img_array)
        predicted_number = np.argmax(predictions[0])
        
        return predicted_number

    def extract_numbers(self,frame):
        first_block = self.price_block(frame)
        w = self.en_x
        block1 = first_block[:, w:w + self.num_width]
        w += self.num_width
        block2 = first_block[:, w:w + self.num_width]
        if self.almost_equal(self.get_pixels(block1), self.missing_digit_id) or self.almost_equal(self.get_pixels(block2), self.missing_digit_id):
            log.info("Item less than $100.00 ...")
            self.buy_item()
            log.error("Bug detected, restarting ...")
            safe_exit()
        else:
            if self.check_price(block1) * 10 + self.check_price(block2) < self.max_price:
                log.info(f"Item less than {self.max_price*1000}. Buying item ...")
                self.buy_item()
                log.error("Bug detected, restarting ...")
                safe_exit()
        log.info("Item too expensive. Moving on ...")
        self.back_button()

    def runner(self):
        store_home_page(high_res())
        log.info("Starting Auction Loop")
        self.main_page()
        log.error("Bug detected, restarting ...")
        safe_exit()
           
def main():
    environ['CUDA_VISIBLE_DEVICES'] = ''
    model = load_model('./trained/active.h5')

    al = MainLoop(model)
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