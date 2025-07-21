#!/bin/bash
#
#Android ADB Installer 2 for Linux Mint 11 and Ubuntu 11.04
#This version is intended to be an improvement on previous version and give the user more installation options and bring more ease to people that want to develop,
#theme, or just whatever other purpose you may want to use something that is in my script for.
#Script written by @ArchDukeDoug and parts of @_CorCor67's scripting code used with permission as well. 
#Thanks to @BoneyNicole, @tabbwabb, and @animedbz16 for putting up with me and proofreading and/or testing the script.
#Script version: 2.0.0

menu=
until [ "$menu" = "0" ]; do
clear
echo ""
echo "***************************************************"
echo "**            Android ADB Installer 2            **"
echo "**                   Main Menu                   **"
echo "***************************************************"
echo "**  1 - Resolve dependencies   ( Run this first )**"
echo "**  2 - Ubuntu 11.10 users run this second.      **"
echo "**  3 - Install the Android SDK                  **"
echo "**  4 - Install the Android NDK                  **"
echo "**  5 - Install Android Debug Bridge (adb)       **"
echo "**  6 - Create/update the 99-android.rules file  **"
echo "**  7 - Full Install                             **"
echo "**  8 - Full Install (Ubuntu 11.10)              **"
echo "**  9 - Help                                     **"
echo "**  0 - Exit                                     **"
echo "***************************************************"
echo ""
echo -n "Enter selection: "
read menu
echo ""
case $menu in
#############################################################
#############################################################
1 ) 
if [ ! -f /var/lib/apt/lists/archive.canonical.com_dists_oneiric_partner_binary-i386_Packages ]
then sudo -v; sudo add-apt-repository "deb http://archive.canonical.com/ $(lsb_release -s -c) partner"
else echo "repository already exists."
fi
sudo -v; sudo apt-get update
#Determine if the operating system is 32 or 64-bit and then install ia32-libs if necessary.
d=ia32-libs

if [[ `getconf LONG_BIT` = "64" ]]; 

then
    echo "64-bit operating system detected.  Checking to see if $d is installed."

    if [[ $(dpkg-query -f'${Status}' --show $d 2>/dev/null) = *\ installed ]]; then
    	echo "$d already installed."
    else
        echo "Installing now..."
    	sudo -v; sudo apt-get --force-yes -y install $d
    fi
else
	echo "32-bit operating system detected.  Skipping."
fi

#Check if openjdk-7-jre-headless is installed
c=openjdk-7-jre-headless
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if openjdk-7-jre is installed
c=openjdk-7-jdk
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if Eclipse is installed
c=eclipse
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if Python is installed
c=python
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if git-core is installed
c=git-core
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if gnupupg is installed
c=gnupg
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if flex is installed
c=flex
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if bison is installed
c=bison
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if gperf is installed
c=gperf
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if build-essential is installed
c=build-essential
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if zip is installed
c=zip
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if curl is installed
c=curl
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if zlib1g-dev is installed
c=zlib1g-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if libc6-dev is installed
c=libc6-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if lib32ncurses5-dev is installed
c=lib32ncurses5-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if x11proto-core-dev is installed
c=x11proto-core-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if libx11-dev is installed
c=libx11-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if lib32readline5-dev is installed
c=lib32readline5-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if lib32z-dev is installed
c=lib32z-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if libgl1-mesa is installed
c=libgl1-mesa-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if g++-multilib is installed
c=g++-multilib
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if mingw32 is installed
c=mingw32
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if tofrodos is installed
c=tofrodos
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if python-markdown is installed
c=python-markdown
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if libxml2-utils is installed
c=libxml2-utils
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Download and install apktool-install.
if [ ! -f /usr/local/bin/aapt ]; then 
	wget http://android-apktool.googlecode.com/files/apktool-install-linux-r04-brut1.tar.bz2 && tar --wildcards --no-anchored -xjvf apktool-install-linux-r04-brut1.tar.bz2; sudo -v; sudo mv aapt /usr/local/bin/; sudo -v; sudo mv apktool /usr/local/bin/; sudo -v; sudo chmod 777 /usr/local/bin/aapt; sudo -v; sudo chmod 777 /usr/local/bin/apktool; rm apktool-install-linux-r04-brut1.tar.bz2; rm -rf apktool-install-linux-r04-brut1/;
else
    echo "apktool-install already installed to /usr/local/bin.  Skipping."
fi

#Download and install apktool.
if [ ! -f "/usr/local/bin/apktool.jar" ]; then 
	wget http://android-apktool.googlecode.com/files/apktool1.4.3.tar.bz2 && tar --wildcards --no-anchored -xjvf apktool1.4.3.tar.bz2; sudo -v; sudo mv apktool.jar /usr/local/bin/; sudo -v; sudo chmod 777 /usr/local/bin/apktool.jar; rm apktool1.4.3.tar.bz2; rm -rf apktool1.4.3/;
else
    echo "apktool already installed to /usr/local/bin.  Skipping."
fi

#Check for ~/bin
if [ ! -d ~/bin ]
then
    mkdir ~/bin
else
    echo "~/bin already exists.  Skipping."
fi

#Set ~/bin to path
if grep -q /home/$USER/bin /home/$USER/.bashrc;
then
    echo "$HOME/bin set up already."
else
    echo "PATH=$PATH:$HOME/bin" >> $HOME/.bashrc
fi

#Check for repo
if [ ! -f /usr/local/bin/repo ];
then
    wget https://dl-ssl.google.com/dl/googlesource/git-repo/repo; sudo -v; sudo mv repo /usr/local/bin/repo; sudo -v; sudo chmod a+x /usr/local/bin/repo
else
    echo "/usr/local/bin/repo already exists."
fi

;;
#############################################################
#############################################################
2 )
#Create a symlink for libX11
if [ -f "/usr/lib/i386-linux-gnu/libX11.so" ]
then
    sudo -v; sudo rm /usr/lib/i386-linux-gnu/libX11.so; sudo -v; sudo ln -s /usr/lib/i386-linux-gnu/libX11.so.6 /usr/lib/i386-linux-gnu/libX11.so
else
    sudo -v; sudo ln -s /usr/lib/i386-linux-gnu/libX11.so.6 /usr/lib/i386-linux-gnu/libX11.so
fi
echo "Please press enter to continue."
read enter
;;
#############################################################
#############################################################
3 )
#Download and install the Android SDK
if [ ! -d "/usr/local/android-sdk" ]; then
for a in $( wget -qO- http://developer.android.com/sdk/index.html | egrep -o "http://dl.google.com[^\"']*linux.tgz" ); do
wget $a && tar --wildcards --no-anchored -xvzf android-sdk_*-linux.tgz; sudo -v; sudo mv android-sdk-linux /usr/local/android-sdk; sudo -v; sudo chmod 777 -R /usr/local/android-sdk; rm android-sdk_*-linux.tgz;
done
else
echo "Android SDK already installed to /usr/local/android-sdk. Skipping."
fi
;;
#############################################################
#############################################################
4 )
#Download and install the Android NDK.
if [ ! -d "/usr/local/android-ndk" ]; then 
	for b in $(  wget -qO- http://developer.android.com/sdk/ndk/index.html | egrep -o "http://dl.google.com[^\"']*linux-x86.tar.bz2"
 ); do wget $b && tar --wildcards --no-anchored -xjvf android-ndk-*-linux-x86.tar.bz2; sudo -v; sudo mv android-ndk-*/ /usr/local/android-ndk; sudo -v; sudo chmod 777 -R /usr/local/android-ndk; rm android-ndk-*-linux-x86.tar.bz2;
	done
else
    echo "Android NDK already installed to /usr/local/android-ndk.  Skipping."
fi
#Create Symlink for Dalvik Debug Monitor Server (DDMS)
if [ -f /bin/ddms ] 
then
    sudo -v; sudo rm /bin/ddms; sudo -v; sudo ln -s /usr/local/android-sdk/tools/ddms /bin/ddms
else
    sudo -v; sudo ln -s /usr/local/android-sdk/tools/ddms /bin/ddms
fi
;;
#############################################################
#############################################################
5 )
#Create a symlink for Android Debug Bridge (adb)
if [ -f /bin/adb ];
then
    sudo -v; sudo rm /bin/adb; sudo -v; sudo ln -s /usr/local/android-sdk/platform-tools/adb /bin/adb
else
    sudo -v; sudo ln -s /usr/local/android-sdk/platform-tools/adb /bin/adb
fi
#Installing adb!
if [ ! -f "/usr/local/android-sdk/platform-tools/adb" ];
then  
	mkdir $HOME/.android; touch $HOME/.android/androidtool.cfg; echo "sdkman.force.http=true" > $HOME/.android/androidtool.cfg; nohup /usr/local/android-sdk/tools/android update sdk > /dev/null 2>&1
else
echo "Android Debug Bridge already detected."
fi
;;
#############################################################
#############################################################
6 )
#Downloads the latest version of 99-android.rules
wget http://dl.dropbox.com/u/4413349/scripts/99-android.rules
sudo -v; sudo mv -f 99-android.rules /etc/udev/rules.d/
;;
#############################################################
#############################################################
7 )
if [ ! -f /var/lib/apt/lists/archive.canonical.com_dists_oneiric_partner_binary-i386_Packages ]
then sudo -v; sudo add-apt-repository "deb http://archive.canonical.com/ $(lsb_release -s -c) partner"
else echo "repository already exists."
fi
sudo -v; sudo apt-get update
#Determine if the operating system is 32 or 64-bit and then install ia32-libs if necessary.
d=ia32-libs

if [[ `getconf LONG_BIT` = "64" ]]; 

then
    echo "64-bit operating system detected.  Checking to see if $d is installed."

    if [[ $(dpkg-query -f'${Status}' --show $d 2>/dev/null) = *\ installed ]]; then
    	echo "$d already installed."
    else
        echo "Installing now..."
    	sudo -v; sudo apt-get --force-yes -y install $d
    fi
else
	echo "32-bit operating system detected.  Skipping."
fi

#Check if openjdk-7-jre-headless is installed
c=openjdk-7-jre-headless
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if openjdk-7-jre is installed
c=openjdk-7-jdk
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if Eclipse is installed
c=eclipse
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if Python is installed
c=python
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if git-core is installed
c=git-core
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if gnupupg is installed
c=gnupg
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if flex is installed
c=flex
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if bison is installed
c=bison
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if gperf is installed
c=gperf
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if build-essential is installed
c=build-essential
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if zip is installed
c=zip
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if curl is installed
c=curl
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if zlib1g-dev is installed
c=zlib1g-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if libc6-dev is installed
c=libc6-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if lib32ncurses5-dev is installed
c=lib32ncurses5-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if x11proto-core-dev is installed
c=x11proto-core-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if libx11-dev is installed
c=libx11-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if lib32readline5-dev is installed
c=lib32readline5-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if lib32z-dev is installed
c=lib32z-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if libgl1-mesa is installed
c=libgl1-mesa-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if g++-multilib is installed
c=g++-multilib
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if mingw32 is installed
c=mingw32
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if tofrodos is installed
c=tofrodos
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if python-markdown is installed
c=python-markdown
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if libxml2-utils is installed
c=libxml2-utils
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Download and install apktool-install.
if [ ! -f /usr/local/bin/aapt ]; then 
	wget http://android-apktool.googlecode.com/files/apktool-install-linux-r04-brut1.tar.bz2 && tar --wildcards --no-anchored -xjvf apktool-install-linux-r04-brut1.tar.bz2; sudo -v; sudo mv aapt /usr/local/bin/; sudo -v; sudo mv apktool /usr/local/bin/; sudo -v; sudo chmod 777 /usr/local/bin/aapt; sudo -v; sudo chmod 777 /usr/local/bin/apktool; rm apktool-install-linux-r04-brut1.tar.bz2; rm -rf apktool-install-linux-r04-brut1/;
else
    echo "apktool-install already installed to /usr/local/bin.  Skipping."
fi

#Download and install apktool.
if [ ! -f "/usr/local/bin/apktool.jar" ]; then 
	wget http://android-apktool.googlecode.com/files/apktool1.4.3.tar.bz2 && tar --wildcards --no-anchored -xjvf apktool1.4.3.tar.bz2; sudo -v; sudo mv apktool.jar /usr/local/bin/; sudo -v; sudo chmod 777 /usr/local/bin/apktool.jar; rm apktool1.4.3.tar.bz2; rm -rf apktool1.4.3/;
else
    echo "apktool already installed to /usr/local/bin.  Skipping."
fi


#Check for ~/bin
if [ ! -d "~/bin" ]
then
    mkdir ~/bin
else
    echo "~/bin already exists.  Skipping."
fi

#Set ~/bin to path
if grep -q /home/$USER/bin /home/$USER/.bashrc;
then
    echo "$HOME/bin set up already."
else
    echo "PATH=$PATH:$HOME/bin" >> $HOME/.bashrc
fi

#Check for repo
if [ ! -f /usr/local/bin/repo ];
then
    wget https://dl-ssl.google.com/dl/googlesource/git-repo/repo; sudo -v; sudo mv repo /usr/local/bin/repo; sudo -v; sudo chmod a+x /usr/local/bin/repo
else
    echo "/usr/local/bin/repo already exists."
fi

#Download and install the Android SDK
if [ ! -d "/usr/local/android-sdk" ]; then
for a in $( wget -qO- http://developer.android.com/sdk/index.html | egrep -o "http://dl.google.com[^\"']*linux.tgz" ); do
wget $a && tar --wildcards --no-anchored -xvzf android-sdk_*-linux.tgz; sudo -v; sudo mv android-sdk-linux /usr/local/android-sdk; sudo -v; sudo chmod 777 -R /usr/local/android-sdk; rm android-sdk_*-linux.tgz;
done
else
echo "Android SDK already installed to /usr/local/android-sdk. Skipping."
fi

#Download and install the Android NDK.
if [ ! -d "/usr/local/android-ndk" ]; then 
	for b in $(  wget -qO- http://developer.android.com/sdk/ndk/index.html | egrep -o "http://dl.google.com[^\"']*linux-x86.tar.bz2"
 ); do wget $b && tar --wildcards --no-anchored -xjvf android-ndk-*-linux-x86.tar.bz2; sudo -v; sudo mv android-ndk-*/ /usr/local/android-ndk; sudo -v; sudo chmod 777 -R /usr/local/android-ndk; rm android-ndk-*-linux-x86.tar.bz2;
	done
else
    echo "Android NDK already installed to /usr/local/android-ndk.  Skipping."
fi
#Create Symlink for Dalvik Debug Monitor Server (DDMS)
if [ -f /bin/ddms ] 
then
    sudo -v; sudo rm /bin/ddms; sudo -v; sudo ln -s /usr/local/android-sdk/tools/ddms /bin/ddms
else
    sudo -v; sudo ln -s /usr/local/android-sdk/tools/ddms /bin/ddms
fi

#Create a symlink for Android Debug Bridge (adb)
if [ -f /bin/adb ];
then
    sudo -v; sudo rm /bin/adb; sudo -v; sudo ln -s /usr/local/android-sdk/platform-tools/adb /bin/adb
else
    sudo -v; sudo ln -s /usr/local/android-sdk/platform-tools/adb /bin/adb
fi
#Installing adb!
if [ ! -f "/usr/local/android-sdk/platform-tools/adb" ];
then  
	mkdir $HOME/.android; touch $HOME/.android/androidtool.cfg; echo "sdkman.force.http=true" > $HOME/.android/androidtool.cfg; nohup /usr/local/android-sdk/tools/android update sdk > /dev/null 2>&1
else
echo "Android Debug Bridge already detected."
fi
#Downloads the latest version of 99-android.rules
wget http://dl.dropbox.com/u/4413349/scripts/99-android.rules
sudo -v; sudo mv -f 99-android.rules /etc/udev/rules.d/
;;
#############################################################
#############################################################
8 ) 
if [ ! -f /var/lib/apt/lists/archive.canonical.com_dists_oneiric_partner_binary-i386_Packages ]
then sudo -v; sudo add-apt-repository "deb http://archive.canonical.com/ $(lsb_release -s -c) partner"
else echo "repository already exists."
fi
sudo -v; sudo apt-get update
#Determine if the operating system is 32 or 64-bit and then install ia32-libs if necessary.
d=ia32-libs

if [[ `getconf LONG_BIT` = "64" ]]; 

then
    echo "64-bit operating system detected.  Checking to see if $d is installed."

    if [[ $(dpkg-query -f'${Status}' --show $d 2>/dev/null) = *\ installed ]]; then
    	echo "$d already installed."
    else
        echo "Installing now..."
    	sudo -v; sudo apt-get --force-yes -y install $d
    fi
else
	echo "32-bit operating system detected.  Skipping."
fi

#Check if openjdk-7-jre-headless is installed
c=openjdk-7-jre-headless
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if openjdk-7-jre is installed
c=openjdk-7-jdk
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if Eclipse is installed
c=eclipse
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if Python is installed
c=python
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if git-core is installed
c=git-core
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if gnupupg is installed
c=gnupg
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if flex is installed
c=flex
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if bison is installed
c=bison
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if gperf is installed
c=gperf
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if build-essential is installed
c=build-essential
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if zip is installed
c=zip
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if curl is installed
c=curl
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if zlib1g-dev is installed
c=zlib1g-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if libc6-dev is installed
c=libc6-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if lib32ncurses5-dev is installed
c=lib32ncurses5-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if x11proto-core-dev is installed
c=x11proto-core-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if libx11-dev is installed
c=libx11-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if lib32readline5-dev is installed
c=lib32readline5-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if lib32z-dev is installed
c=lib32z-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if libgl1-mesa is installed
c=libgl1-mesa-dev
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if g++-multilib is installed
c=g++-multilib
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if mingw32 is installed
c=mingw32
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if tofrodos is installed
c=tofrodos
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if python-markdown is installed
c=python-markdown
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Check if libxml2-utils is installed
c=libxml2-utils
	echo "checking if $c is installed" 2>&1
if [[ $(dpkg-query -f'${Status}' --show $c 2>/dev/null) = *\ installed ]]; 
then
	echo "$c already installed.  Skipping."
else 
	echo "$c was not found, installing..." 2>&1
	sudo -v; sudo apt-get --force-yes -y install $c 2>/dev/null
fi

#Download and install apktool-install.
if [ ! -f /usr/local/bin/aapt ]; then 
	wget http://android-apktool.googlecode.com/files/apktool-install-linux-r04-brut1.tar.bz2 && tar --wildcards --no-anchored -xjvf apktool-install-linux-r04-brut1.tar.bz2; sudo -v; sudo mv aapt /usr/local/bin/; sudo -v; sudo mv apktool /usr/local/bin/; sudo -v; sudo chmod 777 /usr/local/bin/aapt; sudo -v; sudo chmod 777 /usr/local/bin/apktool; rm apktool-install-linux-r04-brut1.tar.bz2; rm -rf apktool-install-linux-r04-brut1/;
else
    echo "apktool-install already installed to /usr/local/bin.  Skipping."
fi

#Download and install apktool.
if [ ! -f "/usr/local/bin/apktool.jar" ]; then 
	wget http://android-apktool.googlecode.com/files/apktool1.4.3.tar.bz2 && tar --wildcards --no-anchored -xjvf apktool1.4.3.tar.bz2; sudo -v; sudo mv apktool.jar /usr/local/bin/; sudo -v; sudo chmod 777 /usr/local/bin/apktool.jar; rm apktool1.4.3.tar.bz2; rm -rf apktool1.4.3/;
else
    echo "apktool already installed to /usr/local/bin.  Skipping."
fi


#Check for ~/bin
if [ ! -d "~/bin" ]
then
    mkdir ~/bin
else
    echo "~/bin already exists.  Skipping."
fi

#Set ~/bin to path
if grep -q /home/$USER/bin /home/$USER/.bashrc;
then
    echo "$HOME/bin set up already."
else
    echo "PATH=$PATH:$HOME/bin" >> $HOME/.bashrc
fi

#Check for repo
if [ ! -f /usr/local/bin/repo ];
then
    wget https://dl-ssl.google.com/dl/googlesource/git-repo/repo; sudo -v; sudo mv repo /usr/local/bin/repo; sudo -v; sudo chmod a+x /usr/local/bin/repo
else
    echo "/usr/local/bin/repo already exists."
fi

#Download and install the Android SDK
if [ ! -d "/usr/local/android-sdk" ]; then
for a in $( wget -qO- http://developer.android.com/sdk/index.html | egrep -o "http://dl.google.com[^\"']*linux.tgz" ); do
wget $a && tar --wildcards --no-anchored -xvzf android-sdk_*-linux.tgz; sudo -v; sudo mv android-sdk-linux /usr/local/android-sdk; sudo -v; sudo chmod 777 -R /usr/local/android-sdk; rm android-sdk_*-linux.tgz;
done
else
echo "Android SDK already installed to /usr/local/android-sdk. Skipping."
fi

#Download and install the Android NDK.
if [ ! -d "/usr/local/android-ndk" ]; then 
	for b in $(  wget -qO- http://developer.android.com/sdk/ndk/index.html | egrep -o "http://dl.google.com[^\"']*linux-x86.tar.bz2"
 ); do wget $b && tar --wildcards --no-anchored -xjvf android-ndk-*-linux-x86.tar.bz2; sudo -v; sudo mv android-ndk-*/ /usr/local/android-ndk; sudo -v; sudo chmod 777 -R /usr/local/android-ndk; rm android-ndk-*-linux-x86.tar.bz2;
	done
else
    echo "Android NDK already installed to /usr/local/android-ndk.  Skipping."
fi
#Create Symlink for Dalvik Debug Monitor Server (DDMS)
if [ -f /bin/ddms ] 
then
    sudo -v; sudo rm /bin/ddms; sudo -v; sudo ln -s /usr/local/android-sdk/tools/ddms /bin/ddms
else
    sudo -v; sudo ln -s /usr/local/android-sdk/tools/ddms /bin/ddms
fi

#Create a symlink for Android Debug Bridge (adb)
if [ -f /bin/adb ];
then
    sudo -v; sudo rm /bin/adb; sudo -v; sudo ln -s /usr/local/android-sdk/platform-tools/adb /bin/adb
else
    sudo -v; sudo ln -s /usr/local/android-sdk/platform-tools/adb /bin/adb
fi
#Installing adb!
if [ ! -f "/usr/local/android-sdk/platform-tools/adb" ];
then  
	mkdir $HOME/.android; touch $HOME/.android/androidtool.cfg; echo "sdkman.force.http=true" > $HOME/.android/androidtool.cfg; nohup /usr/local/android-sdk/tools/android update sdk > /dev/null 2>&1
else
echo "Android Debug Bridge already detected."
fi
#Downloads the latest version of 99-android.rules
wget http://dl.dropbox.com/u/4413349/scripts/99-android.rules
sudo -v; sudo mv -f 99-android.rules /etc/udev/rules.d/
#Create a symlink for libX11
if [ -f "/usr/lib/i386-linux-gnu/libX11.so" ]
then
    sudo -v; sudo rm /usr/lib/i386-linux-gnu/libX11.so; sudo -v; sudo ln -s /usr/lib/i386-linux-gnu/libX11.so.6 /usr/lib/i386-linux-gnu/libX11.so
else
    sudo -v; sudo ln -s /usr/lib/i386-linux-gnu/libX11.so.6 /usr/lib/i386-linux-gnu/libX11.so
fi
;;
#############################################################
#############################################################
9 )
clear
echo "1 - This option will resolve the missing dependencies required for the Android SDK, NDK, and Android Debug Bridge (adb) to function properly."
echo "3 - Installs the Android SDK to the location /usr/local/android-sdk"
echo "4 - Installs the Android NDK to the location /usr/local/android/ndk"
echo "5 - Installs the Android Debug Bridge.  Requires Android SDK and Dependencies."
echo "6 - Downloads or updates your 99-android.rules file to /etc/udev/rules.d/ directory"
echo "7 - Automates steps 1-6."
echo "Please press enter to continue."
read enter
;;
#############################################################
#############################################################
0 ) exit ;;
* ) echo "Please choose from the list above"
	esac
done