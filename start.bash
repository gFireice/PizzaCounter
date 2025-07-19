#!/bin/bash
I=`dpkg -s uvicorn | grep "Status" `
if [ -n "$I" ]
then
    uvicorn main:app --reload
else
    sudo apt install uvicorn
fi