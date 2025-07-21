#!/usr/bin/python
#
#
#########################################################################################
#                                     Disclaimer                                        #k
#########################################################################################
# (c) based on Mobile-Sandbox by
# Michael Spreitzenbarth (research@spreitzenbarth.de)
#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
#########################################################################################
#                          Imports  & Global Variables                                  #
#########################################################################################
# python system imports
import csv
import datetime
import hashlib
import os
import re
import subprocess
import ujson as json
import glob
import config
import uuid
import shutil
import zipfile
# from lib.utils import eprint
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

smali_folder = "/scratch1/NOT_BACKED_UP/cavallarogrp/datasets/feature_space/raw_smali"
skip_pattern = re.compile(
    r'\.(png|jpg|jpeg|webp|gif|wav|mp3|mp4|avi|mov|mkv|flv|aac|ogg|m4a|'
    r'txt|pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|tar|gz|bz2|7z|exe|'
    r'ttf|otf|skin|kotlin_module|kotlin_metadata|data)$', re.IGNORECASE
)
skip_dirs = re.compile(
    r'/com/google/|/com/facebook/|/org/apache/|/unknown/|/android/support/|/androidx/|'
    r'/com/google/android/gms/|/com/google/firebase/|/com/admob/|/com/flurry/|'
    r'/com/chartboost/|/com/unity3d/|/com/squareup/|/com/jakewharton/|/com/alibaba/|'
    r'/com/tencent/|/build/|/bin/|/res/|/logs/|/debug/', re.IGNORECASE)

def eprint(*args, **kwargs):
    """Prints the provided arguments to stderr."""
    print(*args, file=sys.stderr, **kwargs)

CC = ''.join(map(chr, list(range(0, 32)) + list(range(127, 160))))
sha = None

LOGO = """
************************************************************************

▓█████▄  ██▀███  ▓█████  ▄▄▄▄    ██▓ ███▄    █  
▒██▀ ██▌▓██ ▒ ██▒▓█   ▀ ▓█████▄ ▓██▒ ██ ▀█   █ 
░██   █▌▓██ ░▄█ ▒▒███   ▒██▒ ▄██▒██▒▓██  ▀█ ██▒
░▓█▄   ▌▒██▀▀█▄  ▒▓█  ▄ ▒██░█▀  ░██░▓██▒  ▐▌██▒
░▒████▓ ░██▓ ▒██▒░▒████▒░▓█  ▀█▓░██░▒██░   ▓██░
 ▒▒▓  ▒ ░ ▒▓ ░▒▓░░░ ▒░ ░░▒▓███▀▒░▓  ░ ▒░   ▒ ▒ 
 ░ ▒  ▒   ░▒ ░ ▒░ ░ ░  ░▒░▒   ░  ▒ ░░ ░░   ░ ▒░
 ░ ░  ░   ░░   ░    ░    ░    ░  ▒ ░   ░   ░ ░ 
   ░       ░        ░  ░ ░       ░           ░ 
 ░                            ░                
 
************************************************************************
"""


#########################################################################################
#                                    Functions                                          #
#########################################################################################
# create Log file
def createLogFile(logDir):
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    logFile = open(logDir + "static.log", "a+")
    logFile.write("\n\n\n")
    logFile.write(LOGO)
    logFile.write("\n\t" + "static analysis")
    logFile.write(
        "\n\t" + str(datetime.datetime.today()).split(' ')[0] + "\t-\t" +
        str(datetime.datetime.today()).split(' ')[1].split('.')[0])
    logFile.write("\n\n")
    return logFile


# make local log entries
def log(logFile, file, message, type):
    if type == 0:
        logFile.write("\n")
        logFile.write(
            "-----------------------------------------------------------------------\n")
        logFile.write("\t" + message + "\n")
        logFile.write(
            "-----------------------------------------------------------------------\n")
    if type == 1:
        logFile.write("\t\t" + file + "\t" + message + "\n")


# log file footer
def closeLogFile(logFile):
    logFile.write("\n\n\n")
    logFile.write(
        "---------------------------------------------------------------------------------------------------------------------------------\n")
    logFile.write("\t (c) static analysis based on mobile sandbox by mspreitz \t\t www.mobile-sandbox.com")
    logFile.close()

def process_file_APIpermissions(file_path, apiCallList):
    apiPermissions = []
    apiCalls = []
    try:
        with open(file_path, 'r') as file:
            smaliFileContent = file.read()
            for apiCall, permission in apiCallList:
                if apiCall in smaliFileContent:
                    permission = permission.strip()
                    if permission and (permission not in apiPermissions):
                        apiPermissions.append(permission)
                    apiCalls.append((apiCall, permission))
    except Exception as e:
        # Return errors instead of printing, to handle later
        return None, [file_path], str(e)
    
    return (apiPermissions, apiCalls), [], None


# get permissions by used API
def checkAPIpermissions(smaliLocation):
    # Precompile regex for faster execution
    

    apiPermissions = []
    apiCalls = []

    error_apk = []

    with open(config.APICALLS) as f:
        apiCallList = [line.strip().split('|') for line in f if '|' in line]

    # fileList = []
    # for dirname, _, filenames in os.walk(smaliLocation):
    #     for filename in filenames:
    #         if not skip_pattern.search(filename):
    #             fileList.append(os.path.join(dirname, filename))
    fileList = [
        os.path.join(dirname, filename)
        for dirname, _, filenames in os.walk(smaliLocation)
        if not skip_dirs.search(dirname)  # 跳过第三方库目录
        for filename in filenames
        if not skip_pattern.search(filename)  # 跳过无关文件类型
        and os.path.getsize(os.path.join(dirname, filename)) > 100  # 文件大小大于 100 字节
    ]

    # Parse every file in file-list and search for every API call in API-Call-List
    for file_path in tqdm(fileList, desc="Processing files"):
        try:
            with open(file_path, 'r') as file:
                smaliFileContent = file.read()
                for apiCall, permission in apiCallList:
                    if apiCall in smaliFileContent:
                        permission = permission.strip()
                        if permission and (permission not in apiPermissions):
                            apiPermissions.append(permission)
                        apiCalls.append((apiCall, permission))
        except Exception as e:
            # Log the exception or pass
            print(f"[WARNING] File {file} has illegal characters in its name: {str(e)}")
            error_apk.append(smaliLocation.split('/')[-1])
            continue
    
    with open("/cs/academic/phd3/xinrzhen/xinran/drebin-feature-extraction/drebin-extractor/src/fail.txt", "a") as fail_file:
            for apk in list(set(error_apk)):
                fail_file.write(apk + "\n")

    return (apiPermissions, apiCalls)


# using baksmali
# def dex2X(tmpDir, dexFile):
#     # baksmali
#     smaliLocation = tmpDir + "smali"
#     os.mkdir(smaliLocation)
#     baksmali = subprocess.Popen(
#         ['java', '-Xmx256M', '-jar', config.BACKSMALI, '-o', smaliLocation, dexFile])
#     baksmali.wait()
#     return smaliLocation

def dex2X(tmpDir, dexFile):
    # 设置smali输出路径
    smaliLocation = os.path.join(tmpDir, "smali")
    # 安全创建输出目录
    os.makedirs(smaliLocation, exist_ok=True)

    # 检查BACKSMALI路径
    assert os.path.exists(config.BACKSMALI), "Baksmali JAR file not found at the specified path."
    # 检查dexFile路径
    assert os.path.exists(dexFile), "DEX file not found at the specified path."
    assert os.path.getsize(dexFile) > 0, "DEX file is empty or corrupted."

    # 调用baksmali进行反编译
    baksmali = subprocess.Popen(
        ['java', '-Xmx1G', '-jar', config.BACKSMALI, '--no-debug-info', '-o', smaliLocation, dexFile],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    stdout, stderr = baksmali.communicate()
    
    # 检查baksmali的执行结果
    if baksmali.returncode != 0:
        print(f"Error occurred while disassembling DEX file: {stderr.decode('utf-8')}")
    else:
        print(f"DEX file disassembled successfully. Output at: {smaliLocation}")
    
    return smaliLocation


# get all used activities
# the first activity in the list is the MAIN activity
def getActivities(sampleFile):
    activities = []
    manifest = subprocess.Popen([config.AAPT, 'dump', 'badging', sampleFile],
                                stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    manifest = manifest.communicate()[0].decode('utf-8').split("\n")
    for line in manifest:
        if "activity" in line:
            try:
                activity = line.split("'")[1].split(".")[-1]
                activity = re.compile('[%s]' % re.escape(CC)).sub('', activity)
                activity = "." + activity
		      #print activity
                activities.append(activity)
            except:
                continue
        else:
            continue
    manifest = subprocess.Popen(
        [config.AAPT, 'd', 'xmltree', sampleFile, 'AndroidManifest.xml'],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE)
    manifest = manifest.communicate()[0].decode('utf-8').split("\n")
    for i, line in enumerate(manifest):
        if "activity" in line:
            try:
                if 'Raw' not in manifest[i+1]:
                    nextLine = manifest[i + 2].split("=")[1].split('"')[1]
                else:
                    nextLine = manifest[i + 1].split("=")[1].split('"')[1]
                #print 'VEDIAMO', nextLine
                nextLine = re.compile('[%s]' % re.escape(CC)).sub('', nextLine)
                if (nextLine not in activities) and (nextLine != ""):
                    activities.append(nextLine)
                else:
                    continue
            except:
                continue
        else:
            continue
    return activities


# get the used features
def getFeatures(logFile, sampleFile):
    appFeatures = []
    sampleInfos = subprocess.Popen([config.AAPT, 'd', 'badging', sampleFile],
                                   stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
    sampleInfos = sampleInfos.communicate()[0].decode('utf-8').split("\n")
    log(logFile, 0, "application features", 0)
    for sampleInfo in sampleInfos:
        if sampleInfo.startswith("uses-feature"):
            sampleFeature = sampleInfo.split("'")[1]
            sampleFeature = re.compile('[%s]' % re.escape(CC)).sub('',
                                                                   sampleFeature)
            log(logFile, "Feature:", sampleFeature, 1)
            if (sampleFeature not in appFeatures) and (sampleFeature != ""):
                appFeatures.append(sampleFeature)
        else:
            continue
    return appFeatures


# get a list of files inside the apk
def getFilesInsideApk(sampleFile):
    appFiles = []
    xml = subprocess.Popen([config.AAPT, 'list', sampleFile],
                           stdout=subprocess.PIPE,
                           stdin=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    xml = xml.communicate()[0].decode('utf-8').split("\n")
    for line in xml:
        try:
            files = line.split("\n")[0]
            files = re.compile('[%s]' % re.escape(CC)).sub('', files)
            if files != "":
                appFiles.append(files)
        except:
            continue
    return appFiles


# get intents
def getIntents(logFile, sampleFile):
    log(logFile, 0, "used intents", 0)
    appIntents = []
    xml = subprocess.Popen(
        [config.AAPT, 'd', 'xmltree', sampleFile, 'AndroidManifest.xml'],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE)
    xml = xml.communicate()[0].decode('utf-8').split("\n")
    for line in xml:
        if "intent" in line:
            try:
                intents = line.split("=")[1].split("\"")[1]
                intents = re.compile('[%s]' % re.escape(CC)).sub('', intents)
                log(logFile, "AndroidManifest", intents, 1)
                appIntents.append(intents)
            except:
                continue
        else:
            continue
    return appIntents


def getNet(sampleFile):
    appNet = []
    xml = subprocess.Popen(
        [config.AAPT, 'd', 'xmltree', sampleFile, 'AndroidManifest.xml'],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE)
    xml = xml.communicate()[0].decode('utf-8').split("\n")
    for line in xml:
        if "android.net" in line:
            try:
                net = line.split("=")[1].split("\"")[1]
                net = re.compile('[%s]' % re.escape(CC)).sub('', net)
                if net != "":
                    appNet.append(net)
            except:
                continue
        else:
            continue
    return appNet


# get the permissions from the manifest
# different from the permissions when using aapt d xmltree sampleFile AndroidManifest.xml ???
def getPermissions(logFile, sampleFile):
    appPermissions = []
    permissions = subprocess.Popen(
        [config.AAPT, 'd', 'permissions', sampleFile],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE)
    permissions = permissions.communicate()[0].decode('utf-8').split("uses-permission: ")
    log(logFile, 0, "granted permissions", 0)
    i = 1
    while i < len(permissions):
        permission = permissions[i].split("\n")[0]
        permission = re.compile('[%s]' % re.escape(CC)).sub('', permission)
        log(logFile, "Permission:", permission, 1)
        i += 1
        if permission != "":
            appPermissions.append(permission)
    return appPermissions


# get providers
def getProviders(logFile, sampleFile):
    log(logFile, 0, "used providers", 0)
    appProviders = []
    xml = subprocess.Popen(
        [config.AAPT, 'd', 'xmltree', sampleFile, 'AndroidManifest.xml'],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE)
    xml = xml.communicate()[0].decode('utf-8').split("\n")
    for line in xml:
        if "provider" in line:
            try:
                provider = line.split("=")[1].split("\"")[1]
                provider = re.compile('[%s]' % re.escape(CC)).sub('', provider)
                log(logFile, "AndroidManifest", provider, 1)
                if appProviders != "":
                    appProviders.append(provider)
            except:
                continue
        else:
            continue
    return appProviders


# get some basic information
def getSampleInfo(logFile, sampleFile):
    global sha
    fp = open(sampleFile, 'rb')
    content = fp.read()
    md5OfNewJob = hashlib.md5(content).hexdigest().upper()
    shaOfNewJob = hashlib.sha256(content).hexdigest().upper()
    sha = shaOfNewJob
    fp.close()
    appInfos = []
    log(logFile, 0, "application infos", 0)
    log(logFile, "sha256:", shaOfNewJob, 1)
    appInfos.append(shaOfNewJob)
    log(logFile, "md5:", md5OfNewJob, 1)
    appInfos.append(md5OfNewJob)
    sampleInfos = subprocess.Popen([config.AAPT, 'd', 'badging', sampleFile],
                                   stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
    sampleInfos = sampleInfos.communicate()[0].decode('utf-8').split("\n")
    i = 0
    while i < len(sampleInfos):
        sampleInfo = sampleInfos[i]
        if sampleInfo.startswith("package: name="):
            sampleLable = sampleInfo.split("name=")[1].split("'")[1]
            appInfos.append(sampleLable.encode('ascii', 'replace'))
            log(logFile, "Label:", sampleLable, 1)
            break
        else:
            if i == (len(sampleInfos) - 1):
                sampleLable = "NO_LABEL"
                log(logFile, "Label:", "no application label specified", 1)
                appInfos.append(sampleLable)
                break
            else:
                i = i + 1
    i = 0
    while i < len(sampleInfos):
        sampleInfo = sampleInfos[i]
        if sampleInfo.startswith("sdkVersion"):
            sampleSdkVersion = sampleInfo.split("'")[1]
            log(logFile, "SDK version:", sampleSdkVersion, 1)
            appInfos.append(sampleSdkVersion)
            break
        else:
            if i == (len(sampleInfos) - 1):
                sampleSdkVersion = "0"
                log(logFile, "SDK version:", "none specified", 1)
                appInfos.append(sampleSdkVersion)
                break
            else:
                i = i + 1
    apkName = str(sampleFile).split("/")[-1]
    appInfos.append(apkName)
    return appInfos


# get services and receiver
def getServicesReceivers(logFile, sampleFile):
    log(logFile, 0, "used services and receivers", 0)
    servicesANDreceiver = []
    manifest = subprocess.Popen(
        [config.AAPT, 'd', 'xmltree', sampleFile, 'AndroidManifest.xml'],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE)
    manifest = manifest.communicate()[0].decode('utf-8').split("\n")
    for i, line in enumerate(manifest):
        if "service" in line:
            try:
                nextLine = manifest[i + 1].split("=")[1].split('"')[1]
                nextLine = re.compile('[%s]' % re.escape(CC)).sub('', nextLine)
                log(logFile, "AndroidManifest", nextLine, 1)
                if (nextLine not in servicesANDreceiver) and (nextLine != ""):
                    servicesANDreceiver.append(
                        nextLine)
            except:
                continue
        else:
            continue
    for i, line in enumerate(manifest):
        if "receiver" in line:
            try:
                nextLine = manifest[i + 1].split("=")[1].split('"')[1]
                nextLine = re.compile('[%s]' % re.escape(CC)).sub('', nextLine)
                log(logFile, "AndroidManifest", nextLine, 1)
                if (nextLine not in servicesANDreceiver) and (nextLine != ""):
                    servicesANDreceiver.append(
                        nextLine)
            except:
                continue
        else:
            continue
    return servicesANDreceiver



# get services and receiver
def getServices(logFile, sampleFile):
    log(logFile, 0, "used services", 0)
    servicesANDreceiver = []
    manifest = subprocess.Popen(
        [config.AAPT, 'd', 'xmltree', sampleFile, 'AndroidManifest.xml'],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE)
    manifest = manifest.communicate()[0].decode('utf-8').split("\n")
    for i, line in enumerate(manifest):
        if "service" in line:
            try:
                nextLine = manifest[i + 1].split("=")[1].split('"')[1]
                nextLine = re.compile('[%s]' % re.escape(CC)).sub('', nextLine)
                log(logFile, "AndroidManifest", nextLine, 1)
                if (nextLine not in servicesANDreceiver) and (nextLine != ""):
                    servicesANDreceiver.append(
                        nextLine)
            except:
                continue
        else:
            continue
    return servicesANDreceiver


# get services and receiver
def getReceivers(logFile, sampleFile):
    log(logFile, 0, "used receivers", 0)
    servicesANDreceiver = []
    manifest = subprocess.Popen(
        [config.AAPT, 'd', 'xmltree', sampleFile, 'AndroidManifest.xml'],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE)
    manifest = manifest.communicate()[0].decode('utf-8').split("\n")
    for i, line in enumerate(manifest):
        if "receiver" in line:
            try:
                nextLine = manifest[i + 1].split("=")[1].split('"')[1]
                nextLine = re.compile('[%s]' % re.escape(CC)).sub('', nextLine)
                log(logFile, "AndroidManifest", nextLine, 1)
                if (nextLine not in servicesANDreceiver) and (nextLine != ""):
                    servicesANDreceiver.append(
                        nextLine)
            except:
                continue
        else:
            continue
    return servicesANDreceiver


# parsing smali-output for suspicious content
def parseSmaliCalls(logFile, smaliLocation):
    log(logFile, 0, "potentially suspicious api-calls", 0)
    dangerousCalls = []
    # skip_pattern = re.compile(r'\.(png|jpg|jpeg|webp|gif|wav|jar|swf|mp4|avi|mov|mkv|flv|mp3|aac|ogg|m4a|txt|pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|tar|gz|bz2|7z|exe|dll|so|bin|html|css|js|json|db|sql|apk)$', re.IGNORECASE)

    # create file-list of directory
    fileList = [
        os.path.join(dirname, filename)
        for dirname, _, filenames in os.walk(smaliLocation)
        if not skip_dirs.search(dirname)  # 跳过第三方库目录
        for filename in filenames
        if not skip_pattern.search(filename)  # 跳过无关文件类型
        and os.path.getsize(os.path.join(dirname, filename)) > 100  # 文件大小大于 100 字节
    ]
    # fileList = []
    # for dirname, _, filenames in os.walk(smaliLocation):
    #     for filename in filenames:
    #         if not skip_pattern.search(filename):
    #             fileList.append(os.path.join(dirname, filename))
    print(f"Found {len(fileList)} files in the smali directory.")
        
    # parse every file in file-list
    for file in fileList:
        try:
            file = re.compile('[%s]' % re.escape(CC)).sub('', file)
            smaliFile = open(file).readlines()
            i = 0
            for line in smaliFile:
                i += 1
                if "Cipher" in line:
                    try:
                        prevLine = \
                            smaliFile[smaliFile.index(line) - 2].split("\n")[
                                0].split('"')[1]
                        dangerous_call = f"Cipher({prevLine})"
                        # print(f"Found Cipher call in {dangerous_call}")
                        # log(logFile, file + ":" + str(i), line.split("\n")[0],
                        #     # 1)
                        if dangerous_call in dangerousCalls:
                        # if "Cipher(" + prevLine + ")" in dangerousCalls:
                            continue
                        else:
                            dangerousCalls.append("Cipher(" + prevLine + ")")
                    except:
                        continue
                # only for logging !
                if "crypto" in line:
                    try:
                        line = line.split("\n")[0]
                        # log(logFile, file + ":" + str(i), line, 1)
                    except:
                        continue
                if "Ljava/net/HttpURLconnection;->setRequestMethod(Ljava/lang/String;)" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "HTTP GET/POST (Ljava/net/HttpURLconnection;->setRequestMethod(Ljava/lang/String;))" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "HTTP GET/POST (Ljava/net/HttpURLconnection;->setRequestMethod(Ljava/lang/String;))")
                if "Ljava/net/HttpURLconnection" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "HttpURLconnection (Ljava/net/HttpURLconnection)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "HttpURLconnection (Ljava/net/HttpURLconnection)")
                if "getExternalStorageDirectory" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "Read/Write External Storage" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("Read/Write External Storage")
                if "getSimCountryIso" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "getSimCountryIso" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("getSimCountryIso")
                if "execHttpRequest" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "execHttpRequest" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("execHttpRequest")
                if "Lorg/apache/http/client/methods/HttpPost" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "HttpPost (Lorg/apache/http/client/methods/HttpPost)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "HttpPost (Lorg/apache/http/client/methods/HttpPost)")
                if "Landroid/telephony/SmsMessage;->getMessageBody" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "readSMS (Landroid/telephony/SmsMessage;->getMessageBody)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "readSMS (Landroid/telephony/SmsMessage;->getMessageBody)")
                if "sendTextMessage" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "sendSMS" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("sendSMS")
                if "getSubscriberId" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "getSubscriberId" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("getSubscriberId")
                if "getDeviceId" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "getDeviceId" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("getDeviceId")
                if "getPackageInfo" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "getPackageInfo" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("getPackageInfo")
                if "getSystemService" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "getSystemService" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("getSystemService")
                if "getWifiState" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "getWifiState" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("getWifiState")
                if "system/bin/su" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "system/bin/su" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("system/bin/su")
                if "setWifiEnabled" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "setWifiEnabled" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("setWifiEnabled")
                if "setWifiDisabled" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "setWifiDisabled" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("setWifiDisabled")
                if "getCellLocation" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "getCellLocation" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("getCellLocation")
                if "getNetworkCountryIso" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "getNetworkCountryIso" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("getNetworkCountryIso")
                if "SystemClock.uptimeMillis" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "SystemClock.uptimeMillis" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("SystemClock.uptimeMillis")
                if "getCellSignalStrength" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "getCellSignalStrength" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("getCellSignalStrength")
                if "Landroid/os/Build;->BRAND:Ljava/lang/String" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "Access Device Info (Landroid/os/Build;->BRAND:Ljava/lang/String)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "Access Device Info (Landroid/os/Build;->BRAND:Ljava/lang/String)")
                if "Landroid/os/Build;->DEVICE:Ljava/lang/String" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "Access Device Info (Landroid/os/Build;->DEVICE:Ljava/lang/String)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "Access Device Info (Landroid/os/Build;->DEVICE:Ljava/lang/String)")
                if "Landroid/os/Build;->MODEL:Ljava/lang/String" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "Access Device Info (Landroid/os/Build;->MODEL:Ljava/lang/String)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "Access Device Info (Landroid/os/Build;->MODEL:Ljava/lang/String)")
                if "Landroid/os/Build;->PRODUCT:Ljava/lang/String" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "Access Device Info (Landroid/os/Build;->PRODUCT:Ljava/lang/String)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "Access Device Info (Landroid/os/Build;->PRODUCT:Ljava/lang/String)")
                if "Landroid/os/Build;->FINGERPRINT:Ljava/lang/String" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "Access Device Info (Landroid/os/Build;->FINGERPRINT:Ljava/lang/String)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "Access Device Info (Landroid/os/Build;->FINGERPRINT:Ljava/lang/String)")
                if "adb_enabled" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "Check if adb is enabled" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("Check if adb is enabled")
                # used by exploits and bad programers
                if "Ljava/io/IOException;->printStackTrace" in line:
                    # log(logFile, file + ":" + str(i), line.split("\n")[0], 1)
                    if "printStackTrace" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("printStackTrace")
                if "Ljava/lang/Runtime;->exec" in line:
                    # log(logFile, file + ":" + str(i), line, 1)
                    if "Execution of external commands (Ljava/lang/Runtime;->exec)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "Execution of external commands (Ljava/lang/Runtime;->exec)")
                if "Ljava/lang/System;->loadLibrary" in line:
                    # log(logFile, file + ":" + str(i), line, 1)
                    if "Loading of external Libraries (Ljava/lang/System;->loadLibrary)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "Loading of external Libraries (Ljava/lang/System;->loadLibrary)")
                if "Ljava/lang/System;->load" in line:
                    # log(logFile, file + ":" + str(i), line, 1)
                    if "Loading of external Libraries (Ljava/lang/System;->load)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "Loading of external Libraries (Ljava/lang/System;->load)")
                if "Ldalvik/system/DexClassLoader;" in line:
                    # log(logFile, file + ":" + str(i), line, 1)
                    if "Loading of external Libraries (Ldalvik/system/DexClassLoader;)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "Loading of external Libraries (Ldalvik/system/DexClassLoader;)")
                if "Ldalvik/system/SecureClassLoader;" in line:
                    # log(logFile, file + ":" + str(i), line, 1)
                    if "Loading of external Libraries (Ldalvik/system/SecureClassLoader;)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "Loading of external Libraries (Ldalvik/system/SecureClassLoader;)")
                if "Ldalvik/system/PathClassLoader;" in line:
                    # log(logFile, file + ":" + str(i), line, 1)
                    if "Loading of external Libraries (Ldalvik/system/PathClassLoader;)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "Loading of external Libraries (Ldalvik/system/PathClassLoader;)")
                if "Ldalvik/system/BaseDexClassLoader;" in line:
                    # log(logFile, file + ":" + str(i), line, 1)
                    if "Loading of external Libraries (Ldalvik/system/BaseDexClassLoader;)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "Loading of external Libraries (Ldalvik/system/BaseDexClassLoader;)")
                if "Ldalvik/system/URLClassLoader;" in line:
                    # log(logFile, file + ":" + str(i), line, 1)
                    if "Loading of external Libraries (Ldalvik/system/URLClassLoader;)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "Loading of external Libraries (Ldalvik/system/URLClassLoader;)")
                if "android/os/Exec" in line:
                    # log(logFile, file + ":" + str(i), line, 1)
                    if "Execution of native code (android/os/Exec)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append(
                            "Execution of native code (android/os/Exec)")
                if "Base64" in line:
                    # log(logFile, file + ":" + str(i), line, 1)
                    if "Obfuscation(Base64)" in dangerousCalls:
                        continue
                    else:
                        dangerousCalls.append("Obfuscation(Base64)")
                else:
                    continue
        except:
            # eprint(f"[WARNING] File {file} has illegal characters in its name!")
            continue
    return dangerousCalls

def process_file_smaliurl(file):
    urls = []
    try:
        with open(file, 'r') as smaliFile:
            for line in smaliFile:
                # Search for URLs
                try:
                    urlPattern = re.search(
                        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                        line).group()
                    if urlPattern and urlPattern not in urls:
                        urls.append(urlPattern)
                except:
                    pass
                
                # Search for IP addresses
                try:
                    ips = re.search(
                        '(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})',
                        line).group()
                    if ips and ips not in urls:
                        urls.append(ips)
                except:
                    pass
    except Exception as e:
        pass  # Handle file-related exceptions, log if necessary
    return urls


# parsing smali-output for URL's and IP's
def parseSmaliURL(smaliLocation):
    url = []
    # create file-list of directory
    fileList = []
    # log(logFile, 0, "URL's and IP's inside the code", 0)
    # skip_pattern = re.compile(r'\.(png|jpg|jpeg|webp|gif|wav|jar|swf|mp4|avi|mov|mkv|flv|mp3|aac|ogg|m4a|txt|pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|tar|gz|bz2|7z|exe|dll|so|bin|html|css|js|json|db|sql|apk)$', re.IGNORECASE)
    
    fileList = []
    for dirname, _, filenames in os.walk(smaliLocation):
        for filename in filenames:
            if not skip_pattern.search(filename):
                fileList.append(os.path.join(dirname, filename))
    # parse every file in file-list
    for file in tqdm(fileList, desc="Processing files"):
        try:
            i = 0
            smaliFile = open(file).readlines()
            for line in smaliFile:
                try:
                    urlPattern = re.search(
                        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                        line).group()
                    # log(logFile, file + ":" + str(i), urlPattern, 1)
                    if (urlPattern not in url) and (urlPattern != ""):
                        url.append(urlPattern)
                    else:
                        continue
                except:
                    continue
                try:
                    ips = re.search(
                        '(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})',
                        line).group()
                    # log(logFile, file + ":" + str(i), ips, 1)
                    if (ips not in url) and (ips != ""):
                        url.append(ips)
                    else:
                        continue
                except:
                    continue
                i += 1
        except:
            # eprint(f"[WARNING] File {file} has illegal characters in its name!")
            continue
    return url
    # with ThreadPoolExecutor() as executor:
    #     futures = {executor.submit(process_file_smaliurl, file): file for file in fileList}
        
    #     # Process results as they complete
    #     for future in tqdm(as_completed(futures), total=len(fileList), desc="Processing files"):
    #         result = future.result()  # Get the result from each file processing
    #         all_urls.extend(result)  # Add the found URLs and IPs to the main list

    # # Return the aggregated URL list
    # return all_urls


# unpack the sample apk-file
# def unpackSample(tmpDir, sampleFile):
#     os.system("unzip " + "-o -q -d " + tmpDir + " " + sampleFile)
#     return tmpDir


def unpackSample(tmpDir, sampleFile):
    # 确保临时目录存在
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)
    
    try:
        # 使用zipfile模块解压
        with zipfile.ZipFile(sampleFile, 'r') as zip_ref:
            # 提取所有文件到目标目录
            zip_ref.extractall(tmpDir)
        print(f"解压成功：{sampleFile} -> {tmpDir}")
    except zipfile.BadZipFile:
        print(f"错误：无法解压 {sampleFile}，该文件可能不是有效的APK文件。")
    except Exception as e:
        print(f"解压过程中出错：{e}")

    return tmpDir


# check for Ad-Networks
def detect(smaliLocation):
    with open(config.ADSLIBS, 'Ur') as f:
        smaliPath = list(tuple(rec) for rec in csv.reader(f, delimiter=';'))
    fileList = list()
    detectedAds = list()
    for dirname, dirnames, filenames in os.walk(smaliLocation):
        for filename in filenames:
            fileList.append(os.path.join(dirname, filename))
    for path in smaliPath:
        adPath = str(path[1])
        for file in fileList:
            if adPath in file:
                if (str(path[0]) not in detectedAds) and (str(path[0]) != ""):
                    detectedAds.append(str(path[0]))
                else:
                    continue
            else:
                continue
    return detectedAds


# create JSON file
def createOutput(workingDir, appNet, appProviders, appPermissions, appFeatures,
                 appIntents, services, receivers, 
                 dangerousCalls, appUrls, appInfos, apiPermissions, apiCalls,
                 appFiles, appActivities, print_output=False, alt_format=False):
    print_output = False
    output = dict()
    output['md5'] = appInfos[1]
    output['sha256'] = appInfos[0]
    output['package_name'] = appInfos[2]
    output['apk_name'] = appInfos[4]
    output['sdk_version'] = appInfos[3]
    output['app_permissions'] = appPermissions
    output['api_permissions'] = apiPermissions
    output['api_calls'] = apiCalls
    output['features'] = appFeatures
    output['intents'] = appIntents
    output['activities'] = appActivities
    output['services'] = services
    output['receivers'] = receivers
    output['interesting_calls'] = dangerousCalls
    output['urls'] = appUrls
    output['networks'] = appNet
    output['providers'] = appProviders
    output['included_files'] = appFiles
    # output['detected_ad_networks'] = detectedAds
    # save the JSON dict to a file for later use
    if not os.path.exists(workingDir):
        os.mkdir(workingDir)

    # if not os.path.exists(os.path.join(workingDir, 'results')):
    #     os.mkdir(os.path.join(workingDir, 'results'))

    # run_id = '{}drebin-{}@{}'.format(sha, str(uuid.uuid4())[:6],
    #                                  datetime.datetime.utcnow().strftime(
    #                                      '%Y-%m-%dT%H:%M:%SZ'))

    if alt_format:
        output = report_to_feature_vector_alternative(output)
    else:
        output = report_to_feature_vector(output)

    if print_output:
        print("print result: ",json.dumps(output))
    else:
        outpath = os.path.join(workingDir, appInfos[0] + '.json')
        print(("saving results at {}...".format(outpath)))
        jsonFileName = outpath
        jsonFile = open(jsonFileName, "a+")
        jsonFile.write(json.dumps(output))
        jsonFile.close()


# def report_to_feature_vector(report):
#     output = {'sha256': report['sha256']}
#     valid_keywords = ['intents',
#                       'features',
#                       'urls',
#                       'api_calls',
#                       'interesting_calls',
#                       'app_permissions',
#                       'api_permissions',
#                       'activities',
#                       'services',
#                       'receivers',
#                       'providers']

#     def key_fmt(k, val):
#         return '{}::{}'.format(k, val.strip()).replace('.', '_')

#     for k, values in list(report.items()):
#         # TODO: Find out what keywords_mapping is
#         if k in valid_keywords:

#             if k == 'api_calls':
#                 for val in values:
#                     if val[0].strip() != '':
#                         line = key_fmt(k, val[0])
#                         output[line] = 1

#             elif k == 'interesting_calls':
#                 for val in values:
#                     if 'HttpPost' in val:
#                         line = key_fmt(k, val.split(' ')[0])
#                         output[line] = 1
#                     elif '(' in val and ';' in val:
#                         pass
#                     elif val.strip() == '':
#                         pass
#                     elif 'Check if adb is enabled' in val:
#                         pass
#                     else:
#                         line = key_fmt(k, val)
#                         output[line] = 1

#             else:
#                 for val in values:
#                     if val.strip() != '':
#                         line = key_fmt(k, val)
#                         output[line] = 1

#     return output


"""
Return dictionary with features considered by Drebin
TODO: Merge with method above
"""
def report_to_feature_vector_alternative(report):
    output = {'sha256': report['sha256']}
    valid_keywords = ['intents',
                      'features',
                      'urls',
                      'api_calls',
                      'interesting_calls',
                      'app_permissions',
                      'api_permissions',
                      'activities',
                      'services',
                      'receivers',
                      'providers']

    def process_feature(val):
        val = val.decode('utf-8') if isinstance(val, bytes) else val
        return val.strip().replace(".", "_").replace("'", "").replace("name=", "")

    for k, values in list(report.items()):
        # TODO: Find out what keywords_mapping is
        if k in valid_keywords:

            feature_list = list()
            if k == 'api_calls':
                for val in values:
                    if val[0].strip() != '':
                        feature = process_feature(val[0])
                        feature_list.append(feature)

            elif k == 'interesting_calls':
                for val in values:
                    if 'HttpPost' in val:
                        feature = process_feature(val.split(' ')[0])
                        feature_list.append(feature)
                    elif '(' in val and ';' in val:
                        pass
                    elif val.strip() == '':
                        pass
                    elif 'Check if adb is enabled' in val:
                        pass
                    else:
                        feature = process_feature(val)
                        feature_list.append(feature)
            else:
                for val in values:
                    if val.strip() != '':
                        feature = process_feature(val)
                        feature_list.append(feature)

            output[k] = list(set(feature_list))

    return output
        
def report_to_feature_vector(report):
    # Initialize an output dictionary with the sha256 as the key
    output = {}
    sha256_key = report['sha256']
    
    # Initialize a dictionary to store the feature vector for the given sha256
    feature_vector = {}

    valid_keywords = ['intents',
                      'features',
                      'urls',
                      'api_calls',
                      'interesting_calls',
                      'app_permissions',
                      'api_permissions',
                      'activities',
                      'services',
                      'receivers',
                      'providers']

    def key_fmt(k, val):
        return '{}::{}'.format(k, val.strip()).replace('.', '_')

    for k, values in report.items():
        if k in valid_keywords:
            if k == 'api_calls':
                for val in values:
                    if val[0].strip() != '':
                        line = key_fmt(k, val[0])
                        feature_vector[line] = 1

            elif k == 'interesting_calls':
                for val in values:
                    if 'HttpPost' in val:
                        line = key_fmt(k, val.split(' ')[0])
                        feature_vector[line] = 1
                    elif '(' in val and ';' in val:
                        pass
                    elif val.strip() == '':
                        pass
                    elif 'Check if adb is enabled' in val:
                        pass
                    else:
                        line = key_fmt(k, val)
                        feature_vector[line] = 1

            else:
                for val in values:
                    if val.strip() != '':
                        line = key_fmt(k, val)
                        feature_vector[line] = 1

    # Add the feature vector to the output dictionary with sha256 as the key
    output[sha256_key] = feature_vector
    return output



def createTemporaryDir():
    tmp_dir = config.TMPDIR + str(uuid.uuid4())
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"Temporary directory created: {tmp_dir}")
    return tmp_dir

def apk2dext(sampleFile,dexDir):

    apkfile = zipfile.ZipFile(sampleFile,'r')
    apkname = str(apkfile.filename).split('/')[-1][:-4]

    if not os.path.exists(dexDir):
        os.makedirs(dexDir, exist_ok=True)
 
    if os.path.isdir('dex') == False:
        os.mkdir('dex')
    for tempfile in apkfile.namelist(): 
        if tempfile.endswith('.dex'):
            dexfilename = apkname + '.dex'   
            print('dexfile:',dexfilename)
            f = open(dexDir + '/'+ dexfilename,'wb+')
            f.write(apkfile.read(tempfile))


#########################################################################################
#                                  MAIN PROGRAMM                                        #
#########################################################################################


def run(sampleFile, workingDir, print_output,  year, month, alt_format):

    sha256 = sampleFile.split('/')[-1].split('.')[0]
    workingDir = workingDir if workingDir.endswith('/') else workingDir + '/'
    # function calls
    logFilepath = "/cs/academic/phd3/xinrzhen/xinran/drebin-feature-extraction/drebin-extractor"
    print(f"sampleFile: {sampleFile}")
    logFile = createLogFile(logFilepath)
    # temporaryDir = createTemporaryDir()
    # temporaryDir = '/scratch1/NOT_BACKED_UP/cavallarogrp/datasets/processed_dataset/Androzoo_info/drebin_feature_22_23/temp'
    # dexFolder = f'/scratch1/NOT_BACKED_UP/cavallarogrp/datasets/feature_space/raw_dex/{year}-{month}'
    dexFolder = '/scratch1/NOT_BACKED_UP/cavallarogrp/datasets/feature_space/raw_dex/new_drebin'
    dexDir = os.path.join(dexFolder, sha256+'.dex')
    # print("unpacking sample...")
    # unpackLocation = unpackSample(temporaryDir, sampleFile)
    # apk2dext(sampleFile,dexDir)
    print("get Network data...")
    appNet = getNet(sampleFile)
    print("get sample info...")
    appInfos = getSampleInfo(logFile, sampleFile)
    print("get providers...")
    appProviders = getProviders(logFile, sampleFile)
    print("get permissions...")
    appPermissions = getPermissions(logFile, sampleFile)
    print("get activities...")
    appActivities = getActivities(sampleFile)
    print("get features...")
    appFeatures = getFeatures(logFile, sampleFile)
    print("get intents...")
    appIntents = getIntents(logFile, sampleFile)
    print("list files...")
    appFiles = getFilesInsideApk(sampleFile)
    print("get services and receivers...")
    #servicesANDreceiver = getServicesReceivers(logFile, sampleFile)
    services = getServices(logFile, sampleFile)
    receivers = getReceivers(logFile,sampleFile)

    dangerousCalls = []
    appUrls = []
    apiPermissions = []
    apiCalls = []

    # smaliLocation = os.path.join(smali_folder,f"{year}-{month}",sha256)

    # smaliLocation = os.path.join(smali_folder,f"{year}-{month}",sha256)
    smaliLocation = os.path.join(smali_folder,'new_sample',sha256)
    print(f"load smali files from {smaliLocation}")
    # dangerousCalls.extend(parseSmaliCalls(logFile, smaliLocation))
    dangerousCalls.extend(parseSmaliCalls_new(smaliLocation))
    # print "get URLs and IPs..."
    appUrls.extend(parseSmaliURL(smaliLocation))
    # print "check API permissions..."
    perms, calls = checkAPIpermissions(smaliLocation)
    apiPermissions.extend(perms)
    apiCalls.extend(calls)
        # print "check for ad networks..."
        # detectedAds.extend(detect(smaliLocation))
        # print "create json report..."
        # shutil.rmtree(smaliLocation)

    createOutput(workingDir, appNet, appProviders, appPermissions, appFeatures,
                 appIntents, services,receivers, 
                 dangerousCalls, appUrls, appInfos, apiPermissions,
                 apiCalls, appFiles, appActivities, print_output, alt_format)
    # programm and log footer
    # print "close log-file..."
    closeLogFile(logFile)
    # shutil.rmtree(temporaryDir)

def process_file_smaliCalls(file, patterns, dangerousCalls):
    try:
        file = re.compile('[%s]' % re.escape(CC)).sub('', file)
        with open(file, 'r') as smaliFile:
            lines = smaliFile.readlines()
            for line in lines:
                if "Cipher" in line:
                    try:
                        prevLine = lines[lines.index(line) - 2].split("\n")[0].split('"')[1]
                        dangerous_call = f"Cipher({prevLine})"
                        if dangerous_call not in dangerousCalls:
                            dangerousCalls.add(dangerous_call)
                    except IndexError:
                        continue
                for search_term, dangerous_call_template in patterns:
                    if search_term in line:
                        if dangerous_call_template:
                            dangerous_call = dangerous_call_template.format(search_term)
                            dangerousCalls.add(dangerous_call)
                        break
    except Exception as e:
        # Handle exception and move to the next file
        pass
    return dangerousCalls


def parseSmaliCalls_new(smaliLocation):

    dangerousCalls = set()  # Use set for faster membership checks
    # skip_pattern = re.compile(r'\.(png|jpg|jpeg|webp|gif|wav|jar|swf|mp4|avi|mov|mkv|flv|mp3|aac|ogg|m4a|txt|pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|tar|gz|bz2|7z|exe|dll|so|bin|html|css|js|json|db|sql|apk)$', re.IGNORECASE)
    # Create a list of files
    # fileList = [os.path.join(dirname, filename) 
    #             for dirname, _, filenames in os.walk(smaliLocation) 
    #             for filename in filenames 
    #             if not skip_pattern.search(filename)]
    
    fileList = [
        os.path.join(dirname, filename)
        for dirname, _, filenames in os.walk(smaliLocation)
        if not skip_dirs.search(dirname)  # 跳过第三方库目录
        for filename in filenames
        if not skip_pattern.search(filename)  # 跳过无关文件类型
        and os.path.getsize(os.path.join(dirname, filename)) > 100  # 文件大小大于 100 字节
    ]


    # Precompile patterns for efficiency
    patterns = [
        ("crypto", None),
        ("Ljava/net/HttpURLconnection;->setRequestMethod(Ljava/lang/String;)", "HTTP GET/POST (Ljava/net/HttpURLconnection;->setRequestMethod(Ljava/lang/String;))"),
        ("Ljava/net/HttpURLconnection", "HttpURLconnection (Ljava/net/HttpURLconnection)"),
        ("getExternalStorageDirectory", "Read/Write External Storage"),
        ("getSimCountryIso", "getSimCountryIso"),
        ("execHttpRequest", "execHttpRequest"),
        ("Lorg/apache/http/client/methods/HttpPost", "HttpPost (Lorg/apache/http/client/methods/HttpPost)"),
        ("Landroid/telephony/SmsMessage;->getMessageBody", "readSMS (Landroid/telephony/SmsMessage;->getMessageBody)"),
        ("sendTextMessage", "sendSMS"),
        ("getSubscriberId", "getSubscriberId"),
        ("getDeviceId", "getDeviceId"),
        ("getPackageInfo", "getPackageInfo"),
        ("getSystemService", "getSystemService"),
        ("getWifiState", "getWifiState"),
        ("system/bin/su", "system/bin/su"),
        ("setWifiEnabled", "setWifiEnabled"),
        ("setWifiDisabled", "setWifiDisabled"),
        ("getCellLocation", "getCellLocation"),
        ("getNetworkCountryIso", "getNetworkCountryIso"),
        ("SystemClock.uptimeMillis", "SystemClock.uptimeMillis"),
        ("getCellSignalStrength", "getCellSignalStrength"),
        ("Landroid/os/Build;->BRAND:Ljava/lang/String", "Access Device Info (Landroid/os/Build;->BRAND:Ljava/lang/String)"),
        ("Landroid/os/Build;->DEVICE:Ljava/lang/String", "Access Device Info (Landroid/os/Build;->DEVICE:Ljava/lang/String)"),
        ("Landroid/os/Build;->MODEL:Ljava/lang/String", "Access Device Info (Landroid/os/Build;->MODEL:Ljava/lang/String)"),
        ("Landroid/os/Build;->PRODUCT:Ljava/lang/String", "Access Device Info (Landroid/os/Build;->PRODUCT:Ljava/lang/String)"),
        ("Landroid/os/Build;->FINGERPRINT:Ljava/lang/String", "Access Device Info (Landroid/os/Build;->FINGERPRINT:Ljava/lang/String)"),
        ("adb_enabled", "Check if adb is enabled"),
        ("Ljava/io/IOException;->printStackTrace", "printStackTrace"),
        ("Ljava/lang/Runtime;->exec", "Execution of external commands (Ljava/lang/Runtime;->exec)"),
        ("Ljava/lang/System;->loadLibrary", "Loading of external Libraries (Ljava/lang/System;->loadLibrary)"),
        ("Ljava/lang/System;->load", "Loading of external Libraries (Ljava/lang/System;->load)"),
        ("Ldalvik/system/DexClassLoader;", "Loading of external Libraries (Ldalvik/system/DexClassLoader;)"),
        ("Ldalvik/system/SecureClassLoader;", "Loading of external Libraries (Ldalvik/system/SecureClassLoader;)"),
        ("Ldalvik/system/PathClassLoader;", "Loading of external Libraries (Ldalvik/system/PathClassLoader;)"),
        ("Ldalvik/system/BaseDexClassLoader;", "Loading of external Libraries (Ldalvik/system/BaseDexClassLoader;)"),
        ("Ldalvik/system/URLClassLoader;", "Loading of external Libraries (Ldalvik/system/URLClassLoader;)"),
        ("android/os/Exec", "Execution of native code (android/os/Exec)"),
        ("Base64", "Obfuscation(Base64)")
    ]

    # # Process each file
    for file in tqdm(fileList, desc="Processing files"):
        try:
            file = re.compile('[%s]' % re.escape(CC)).sub('', file)
            with open(file, 'r') as smaliFile:
                lines = smaliFile.readlines()
                for line in lines:
                    if "Cipher" in line:
                        try:
                            prevLine = lines[lines.index(line) - 2].split("\n")[0].split('"')[1]
                            dangerous_call = f"Cipher({prevLine})"
                            # print(f"Found dangerous call in Cipher: {dangerous_call}")
                            if dangerous_call not in dangerousCalls:
                                dangerousCalls.add(dangerous_call)
                        except IndexError:
                            continue
                    for search_term, dangerous_call_template in patterns:
                        if search_term in line:
                            if dangerous_call_template:
                                dangerous_call = dangerous_call_template.format(search_term)
                                dangerousCalls.add(dangerous_call)
                            break
        except Exception as e:
            # Handle exception and move to the next file
            continue

    return list(dangerousCalls)  # Return as a list if needed



# if __name__=="__main__":
#     smali_folder = '/scratch1/NOT_BACKED_UP/cavallarogrp/datasets/feature_space/raw_smali/2022-01/A6B4EB45D2BB7FAB3968087159D8A96EFF6C100BED237D975227D79CAF3FE1A0'
#     logFilepath = '/cs/academic/phd3/xinrzhen/xinran/drebin-feature-extraction/drebin-extractor/src/'
#     logFile = createLogFile(logFilepath)

#     new_output = parseSmaliCalls_new(smali_folder)
#     output = parseSmaliCalls(logFile, smali_folder)

#     print(f"New output: {new_output}")
#     print(f"Old output: {output}")

#     diff = set(output) - set(new_output)
#     print(f"Difference: {diff}")