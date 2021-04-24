function n = linecount(fileName)
fid = fopen(fileName,'r');
n = 0;
tline = fgetl(fid);
while ischar(tline)
  tline = fgetl(fid);
  n = n+1;
end

end