#!/bin/bash
# -*- coding: utf-8 -*-

wget https://github.com/kbarbary/sfddata/archive/master.tar.gz
tar xf master.tar.gz
mv sfddata-master herschelhelp_internal/sfd_data
rm -f master.tar.gz
