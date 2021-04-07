#!/bin/bash
dir_path=$1
echo "|INFO:file path : ${dir_path}|"
export LANG="zh_CN.UTF-8"

file_path_list=($(ls ${dir_path}"/data"))


for (( i = 0; i < ${#file_path_list[*]}; i++ )); do
  file_path=${file_path_list[i]}
#  echo "${file_path}"
  file_name=${file_path: 0:-4}
  echo "|${file_name}|"
#  file_name=${file_path}
  if [ ${i} == 0 ]; then
    echo -e "${file_name}">${dir_path}/'file_names.txt'
    echo -e "${file_name} 300 300">${dir_path}/'test_file_names.txt'
  else
    echo -e "${file_name}">>${dir_path}/'file_names.txt'
    echo -e "${file_name} 300 300">>${dir_path}/'test_file_names.txt'
  fi
done