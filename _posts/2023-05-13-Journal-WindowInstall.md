---
layout:       post
title:        "Debug: Install Windows 11"
author:       "Allan"
header-style: text
catalog:      true
mathjax:      true
comments: true
tags:
    - Windows
    - Installation
    - Debug
---
# Remark

I hate windows and Microsoft very much. No way a programmer can build the installation so unfriendly for user who dont have another windows machine in hand.

And OpenAI develop GPT, not Microsoft or Microsoft Research. Microsoft is just a company who buy the technology and use it for their own profit. Just a reminder.

# Background
Due to automatic Windows update, my PC is corrupted due to failure of update. The old driver is deleted while the new driver imposed by Window update is failed. Therefore, all the keyboard or mouse is not working as the driver for unifying USB is no longer recognizable. I cannot even control my PC to enter the safe mode.

Luckily I have wired keyboard in hand. I tried to recover the system. The problem is getting worse as the system is corrupted. The recovery deleted VMD driver and therefore no hard disk is recognized.

I have no choice but to reinstall Windows 11. However, the installation is so unfriendly that I have to spend 2 days to figure out how to install it.

# VMD driver and support

In the old configuration, VMD is turned on, which is provided by Intel to support hot plug of M.2 SSD. However, the driver is deleted by recovery. Therefore, I cannot see any hard disk in the installation page. 

The primary solution is to load the drive from USB on installation page, which require the drive "F6" to load the driver. However, Intel does not provide the zip file of F6. Instead they provide a RSTSetup.exe which is not executable by Linux or MacOS. Good Job Intel. Although the issue is raised to [Intel forum](https://www.intel.com/content/www/us/en/support/articles/000058724/memory-and-storage/intel-optane-memory.html), no practical solution provided.

They suggest a solution that is to use the [RSTSetup.exe](https://www.intel.com/content/www/us/en/download/19512/intel-rapid-storage-technology-driver-installation-software-with-intel-optane-memory-10th-and-11th-gen-platforms.html) to install the driver from another Windows machine to the USB. Then the driver can be loaded in the installation page.

## Solution
Luckily, I can turn off the VMD support in exchange to recognition of hard disk. The solution is to enter the BIOS and turn off the VMD support. Then the hard disk can be recognized in the installation page. However, you need to erase the data in the old hard disks as the partition table with and without VMD is different.

# Creation of Windows 11 USB
In MacOS environment, the creation of Windows 11 USB is not easy. The official solution is to use the [Media Creation Tool](https://www.microsoft.com/en-us/software-download/windows11) to create the USB. However, the tool is only available in Windows environment.

The reason is that the Windows 11 ISO file is larger than 4GB. Therefore, the USB is required to be formatted as NTFS. However, MacOS does not support NTFS format. Therefore, the USB cannot be created in MacOS.

So you need two USB to install the windows. The step is [as follows](https://gist.github.com/bmatcuk/fda5ab0fb127e9fd62eaf43e845a51c3?permalink_comment_id=3579269):


1. Get TWO usb sticks, and format using macOS Disk Utility. First >2GB, FAT32, Master Boot Record (MBR). Second >8GB, formatted as exFAT, MBR.
2. Download the Windows 10/11 ISO. Open on the Mac desktop.
3. Copy everything EXCEPT “sources” folder onto FAT32 USB (drag and drop).
4. On the same USB, create a folder called “sources”, and copy into it the one file “boot.wim” from the “sources” folder in the ISO
5. Copy everything from the ISO onto the exFAT USB. It seems not to matter that some materials will appear on both USBs.
6. Plug both USBs into the PC.
7. The PC was able to boot from the FAT32 USB; and it found the install.wim file (and whatever else it needed) from the exFAT USB without any additional voodoo, and completed the install successfully.

The following command is used to burn the ISO file to USB. The command is executed in MacOS.

```bash
diskutil list external
diskutil eraseDisk MS-DOS "WIN_USB1" MBR disk[your disk1 number]
diskutil eraseDisk ExFAT "WIN_USB2" MBR disk[your disk2 number]
hdiutil mount [your ISO file name]
rsync -avh --progress --exclude=sources /Volumes/[your ISO file name]/ /Volumes/WIN_USB1/
mkdir /Volumes/WIN_USB1/sources
cp /Volumes/[your ISO file name]/sources/boot.wim /Volumes/WIN_USB1/sources/

rsync -avh --progress /Volumes/[your ISO file name]/ /Volumes/WIN_USB2
```

To check the disk number, you can use disk utility in MacOS. 


# Installation
