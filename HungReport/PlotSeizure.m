clc
clear all
close all

DataPath = "EOG_record_5.csv";

Data = csvread(DataPath);

rawleft = Data(:,3);
rawright = Data(:,4);

% xrange = []
yrangeraw = [-300 300];

% starttime = 1+2*250;
% endtime = 30*250;

starttime = 1+10.5*250;
endtime = 11.5*250;
id = 24;

T = (1:(endtime-starttime+1))./250;


figure;
subplot(2,1,1)
plot(T,rawleft(starttime:endtime))
ylabel("Voltage (uV)")
% xlim(xrange)
ylim(yrangeraw)

subplot(2,1,2)
plot(T,rawright(starttime:endtime))
ylabel("Voltage (uV)")
% xlim(xrange)
ylim(yrangeraw)


csvwrite("Data/no/no_"+id+".csv",[rawleft(starttime:endtime) rawright(starttime:endtime)])

