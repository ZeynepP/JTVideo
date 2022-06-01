# -*- coding: utf-8 -*-
"""
Created on Mon Dec 06 11:54:04 2016

@author: zpehlivan@ina.fr

It is a test to get all tweets for a query by using advanced search page of twitter.


"""

from selenium import webdriver

import time
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from datetime import datetime, timedelta
import os
import random
import signal
import sys
import glob

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = '127.0.0.1,localhost'


def getDriver(output_dir):
    path = "/usr/bin/geckodriver"
    binary = FirefoxBinary("/usr/bin/firefox")
    profile = webdriver.FirefoxProfile()
    profile.set_preference("network.proxy.type", 1)
    profile.set_preference("network.proxy.http", "firewall.ina.fr")
    profile.set_preference("network.proxy.http_port", 81)
    profile.set_preference("network.proxy.https", "firewall.ina.fr")
    profile.set_preference("network.proxy.https_port", 81)
    profile.set_preference("network.proxy.ssl", "firewall.ina.fr")
    profile.set_preference("network.proxy.ssl_port", 81)
    profile.set_preference("browser.download.dir", output_dir)
    profile.set_preference("browser.download.folderList", 2)
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "image/png")
    profile.set_preference("browser.helperApps.neverAsk.openFile", "image/png")
    profile.set_preference("browser.download.manager.alertOnEXEOpen", False)
    profile.set_preference("browser.download.manager.focusWhenStarting", False)
    profile.set_preference("browser.download.manager.useWindow", False)
    profile.set_preference("browser.download.manager.showAlertOnComplete", False)
    profile.set_preference("browser.download.manager.closeWhenDone", False)
    profile.update_preferences()

    driver = webdriver.Firefox(firefox_profile=profile, executable_path=path, firefox_binary=binary)

    return driver
def getChromeDriver():
    return webdriver.Chrome()

def getImage(driver, url, output, name):

    driver.get(url)
  #  WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, "upload")))
    WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, "input_id")))
    elem = driver.find_element_by_id("input_id")
    elem.send_keys(name)
    time.sleep(1)





if __name__ == "__main__":

    url = "http://localhost:63343/soundshader.github.io/index.html"

   # getImage( url, '', name)
    for i in [0,1,10,11,12]:
        output_dir = "/media/zpehlivan/SAMSUNG/wav_images/" + str(i)

        driver = getDriver(output_dir)
        files = glob.glob("/media/zpehlivan/SAMSUNG/wavs/"+str(i)  + "/*.wav")
        for name in files:
              out_name = name.replace("wavs","wav_images_rect").replace(".wav",".wav.png")
              print(out_name)
              if not os.path.exists(out_name)  :
                print(name, out_name)
                getImage(driver, url, '', name)

    driver.quit()
