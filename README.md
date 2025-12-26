# Docker Image

we use `uniflexai/robo:latest5` by default, which is CentOS-based image.

# rclone install

```
yum install epel-release
yum install man-db man-pages
curl https://rclone.org/install.sh | bash
```

# rclone config

```
root@C.29197730:~$ rclone config
Current remotes:

Name                 Type
====                 ====
gdrive               drive

e) Edit existing remote
n) New remote
d) Delete remote
r) Rename remote
c) Copy remote
s) Set configuration password
q) Quit config
e/n/d/r/c/s/q> n

Enter name for new remote.
name> backblaze

Option Storage.
Type of storage to configure.
Choose a number from below, or type in your own value.
 1 / 1Fichier
   \ (fichier)
 2 / Akamai NetStorage
   \ (netstorage)
 3 / Alias for an existing remote
   \ (alias)
 4 / Amazon S3 Compliant Storage Providers including AWS, Alibaba, ArvanCloud, Ceph, ChinaMobile, Cloudflare, Cubbit, DigitalOcean, Dreamhost, Exaba, FileLu, FlashBlade, GCS, Hetzner, HuaweiOBS, IBMCOS, IDrive, Intercolo, IONOS, Leviia, Liara, Linode, LyveCloud, Magalu, Mega, Minio, Netease, Outscale, OVHcloud, Petabox, Qiniu, Rabata, RackCorp, Rclone, Scaleway, SeaweedFS, Selectel, Servercore, SpectraLogic, StackPath, Storj, Synology, TencentCOS, Wasabi, Zata, Other
   \ (s3)
 5 / Backblaze B2
   \ (b2)
 6 / Better checksums for other remotes
   \ (hasher)
 7 / Box
   \ (box)
 8 / Cache a remote
   \ (cache)
 9 / Citrix Sharefile
   \ (sharefile)
10 / Cloudinary
   \ (cloudinary)
11 / Combine several remotes into one
   \ (combine)
12 / Compress a remote
   \ (compress)
13 / DOI datasets
   \ (doi)
14 / Dropbox
   \ (dropbox)
15 / Encrypt/Decrypt a remote
   \ (crypt)
16 / Enterprise File Fabric
   \ (filefabric)
17 / FTP
   \ (ftp)
18 / FileLu Cloud Storage
   \ (filelu)
19 / Files.com
   \ (filescom)
20 / Gofile
   \ (gofile)
21 / Google Cloud Storage (this is not Google Drive)
   \ (google cloud storage)
22 / Google Drive
   \ (drive)
23 / Google Photos
   \ (google photos)
24 / HTTP
   \ (http)
25 / Hadoop distributed file system
   \ (hdfs)
26 / HiDrive
   \ (hidrive)
27 / ImageKit.io
   \ (imagekit)
28 / In memory object storage system.
   \ (memory)
29 / Internet Archive
   \ (internetarchive)
30 / Jottacloud
   \ (jottacloud)
31 / Koofr, Digi Storage and other Koofr-compatible storage providers
   \ (koofr)
32 / Linkbox
   \ (linkbox)
33 / Local Disk
   \ (local)
34 / Mail.ru Cloud
   \ (mailru)
35 / Mega
   \ (mega)
36 / Microsoft Azure Blob Storage
   \ (azureblob)
37 / Microsoft Azure Files
   \ (azurefiles)
38 / Microsoft OneDrive
   \ (onedrive)
39 / OpenDrive
   \ (opendrive)
40 / OpenStack Swift (Rackspace Cloud Files, Blomp Cloud Storage, Memset Memstore, OVH)
   \ (swift)
41 / Oracle Cloud Infrastructure Object Storage
   \ (oracleobjectstorage)
42 / Pcloud
   \ (pcloud)
43 / PikPak
   \ (pikpak)
44 / Pixeldrain Filesystem
   \ (pixeldrain)
45 / Proton Drive
   \ (protondrive)
46 / Put.io
   \ (putio)
47 / QingCloud Object Storage
   \ (qingstor)
48 / Quatrix by Maytech
   \ (quatrix)
49 / Read archives
   \ (archive)
50 / SMB / CIFS
   \ (smb)
51 / SSH/SFTP
   \ (sftp)
52 / Sia Decentralized Cloud
   \ (sia)
53 / Storj Decentralized Cloud Storage
   \ (storj)
54 / Sugarsync
   \ (sugarsync)
55 / Transparently chunk/split large files
   \ (chunker)
56 / Uloz.to
   \ (ulozto)
57 / Union merges the contents of several upstream fs
   \ (union)
58 / Uptobox
   \ (uptobox)
59 / WebDAV
   \ (webdav)
60 / Yandex Disk
   \ (yandex)
61 / Zoho
   \ (zoho)
62 / iCloud Drive
   \ (iclouddrive)
63 / premiumize.me
   \ (premiumizeme)
64 / seafile
   \ (seafile)
Storage> 5

Option account.
Account ID or Application Key ID.
Enter a value.
account> 00422b3034e57950000000007

Option key.
Application Key.
Enter a value.
key> K0048E++77cRd57ixCYQxOyrpZSaZIc

Option hard_delete.
Permanently delete files on remote removal, otherwise hide files.
Enter a boolean value (true or false). Press Enter for the default (false).
hard_delete> 

Edit advanced config?
y) Yes
n) No (default)
y/n> 

Configuration complete.
Options:
- type: b2
- account: 00422b3034e57950000000007
- key: K0048E++77cRd57ixCYQxOyrpZSaZIc
Keep this "backblaze" remote?
y) Yes this is OK (default)
e) Edit this remote
d) Delete this remote
y/e/d> 

Current remotes:

Name                 Type
====                 ====
backblaze            b2
gdrive               drive

e) Edit existing remote
n) New remote
d) Delete remote
r) Rename remote
c) Copy remote
s) Set configuration password
q) Quit config
e/n/d/r/c/s/q> q
```

# sync data

## upload
```
rclone sync /mnt/sznas/yzf/stereo_data/SceneFlow/ backblaze:/stereo_data/SceneFlow/ \
  --transfers=32 \
  --checkers=32 \
  --buffer-size=32M \
  --fast-list \
  --multi-thread-streams=32 \
  --progress --stats=5s --stats-one-line
```

## download
```
rclone sync backblaze:/stereo_data/SceneFlow/ /mnt/sznas/yzf/stereo_data/SceneFlow/ \
  --transfers=32 \
  --checkers=32 \
  --buffer-size=32M \
  --fast-list \
  --multi-thread-streams=32 \
  --progress --stats=5s --stats-one-line
```
