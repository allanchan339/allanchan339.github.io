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

Or if you have angel on your side, you can reformat the USB with exFat format and burn the ISO file to your USB. 

The following command is used to burn the ISO file to USB. The command is executed in MacOS.

```bash
diskutil list external
diskutil eraseDisk exFat "[Your USB name]" GBT disk[your disk number]
hdiutil mount [your ISO file name]
rsync -avh --progress /Volumes/[your ISO file name]/ /Volumes/[Your USB name]/
diskutil unmount /dev/disk[your disk number]
```

