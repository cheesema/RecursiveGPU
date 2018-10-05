
function RecursiveBenchmark()
addpath('../APRBench/Matlab');


% -------------------------------------------------------------------------
% Local Intensity plot
% -------------------------------------------------------------------------
xx=figure(1);
clf;
hold on;
format_figure(xx);

[~,gV100_2]=plotDataSimple(['../BenchmarkResults/v100/', 'BenchmarkLocalIntensityScaleTestOffset2.h5'], 0, 1);
[~,gV100_6]=plotDataSimple(['../BenchmarkResults/v100/', 'BenchmarkLocalIntensityScaleTestOffset6.h5'], 0, 1);
[~,gP100_2]=plotDataSimple(['../BenchmarkResults/p100/', 'BenchmarkLocalIntensityScaleTestOffset2.h5'], 0, 1);
[~,gP100_6]=plotDataSimple(['../BenchmarkResults/p100/', 'BenchmarkLocalIntensityScaleTestOffset6.h5'], 0, 1);
[~,g1080_2]=plotDataSimple(['../BenchmarkResults/Furiosa/', 'BenchmarkLocalIntensityScaleTestOffset2.h5'], 0, 1);
[~,g1080_6]=plotDataSimple(['../BenchmarkResults/Furiosa/', 'BenchmarkLocalIntensityScaleTestOffset6.h5'], 0, 1);
[~,gTITAN_2]=plotDataSimple(['../BenchmarkResults/ParticulatorOpenMP/', 'BenchmarkLocalIntensityScaleTestOffset2.h5'], 0, 1);
[~,gTITAN_6]=plotDataSimple(['../BenchmarkResults/ParticulatorOpenMP/', 'BenchmarkLocalIntensityScaleTestOffset6.h5'], 0, 1);
[~,g1080ti_2]=plotDataSimple(['../BenchmarkResults/1080ti/', 'BenchmarkLocalIntensityScaleTestOffset2.h5'], 0, 1);
[~,g1080ti_6]=plotDataSimple(['../BenchmarkResults/1080ti/', 'BenchmarkLocalIntensityScaleTestOffset6.h5'], 0, 1);
[~,g1080ti2_2]=plotDataSimple(['../BenchmarkResults/1080ti2/', 'BenchmarkLocalIntensityScaleTestOffset2.h5'], 0, 1);
[~,g1080ti2_6]=plotDataSimple(['../BenchmarkResults/1080ti2/', 'BenchmarkLocalIntensityScaleTestOffset6.h5'], 0, 1);
[cCPU40_2,~]=plotDataSimple(['../BenchmarkResults/Furiosa/', 'BenchmarkLocalIntensityScaleTestOffset2.h5'], 1, 0);
[cCPU40_6,~]=plotDataSimple(['../BenchmarkResults/Furiosa/', 'BenchmarkLocalIntensityScaleTestOffset6.h5'], 1, 0);
[cCPU10_2,~]=plotDataSimple(['../BenchmarkResults/ParticulatorOpenMP/', 'BenchmarkLocalIntensityScaleTestOffset2.h5'], 1, 0);
[cCPU10_6,~]=plotDataSimple(['../BenchmarkResults/ParticulatorOpenMP/', 'BenchmarkLocalIntensityScaleTestOffset6.h5'], 1, 0);
[cCPU_2,~]=plotDataSimple(['../BenchmarkResults/ParticulatorSingleCPU/', 'BenchmarkLocalIntensityScaleTestOffset2.h5'], 1, 0);
[cCPU_6,~]=plotDataSimple(['../BenchmarkResults/ParticulatorSingleCPU/', 'BenchmarkLocalIntensityScaleTestOffset6.h5'], 1, 0);

v100speedup_2=round(mean(cCPU_2 ./ gV100_2));
p100speedup_2=round(mean(cCPU_2 ./ gP100_2));
g1080speedup_2=round(mean(cCPU_2(1:15) ./ g1080_2));
gTTIANspeedup_2=round(mean(cCPU_2 ./ gTITAN_2));
g1080tispeedup_2=round(mean(cCPU_2 ./ g1080ti_2));
g1080ti2speedup_2=round(mean(cCPU_2 ./ g1080ti2_2));

cpu40speedup_2=round(mean(cCPU_2(1:15) ./ cCPU40_2));
cpu10speedup_2=round(mean(cCPU_2 ./ cCPU10_2));

v100speedup_6=round(mean(cCPU_6 ./ gV100_6));
p100speedup_6=round(mean(cCPU_6 ./ gP100_6));
g1080speedup_6=round(mean(cCPU_6(1:15) ./ g1080_6));
gTTIANspeedup_6=round(mean(cCPU_6 ./ gTITAN_6));
g1080tispeedup_6=round(mean(cCPU_2 ./ g1080ti_6));
g1080ti2speedup_6=round(mean(cCPU_2 ./ g1080ti2_6));

cpu40speedup_6=round(mean(cCPU_6(1:15) ./ cCPU40_6));
cpu10speedup_6=round(mean(cCPU_6 ./ cCPU10_6));

axis([-Inf Inf -Inf 2.5]);    

l = legend( { strcat('v100 off=2 speedup=', num2str(v100speedup_2), 'x'),
              strcat('v100 off=6 speedup=', num2str(v100speedup_6), 'x'),
              strcat('p100 off=2 speedup=', num2str(p100speedup_2), 'x'),
              strcat('p100 off=6 speedup=', num2str(p100speedup_6), 'x'),
              strcat('GTX 1080 off=2 speedup=', num2str(g1080speedup_2), 'x'),
              strcat('GTX 1080 off=6 speedup=', num2str(g1080speedup_6), 'x'),
              strcat('TITAN X off=2 speedup=', num2str(gTTIANspeedup_2), 'x'),
              strcat('TITAN X off=6 speedup=', num2str(gTTIANspeedup_6), 'x'),          
              strcat('1080ti X off=2 speedup=', num2str(g1080tispeedup_2), 'x'),
              strcat('1080ti X off=6 speedup=', num2str(g1080tispeedup_6), 'x'),                        
              strcat('1080ti2 X off=2 speedup=', num2str(g1080ti2speedup_2), 'x'),
              strcat('1080ti2 X off=6 speedup=', num2str(g1080ti2speedup_6), 'x'),                        
              strcat('40xCPU off=2 speedup=', num2str(cpu40speedup_2), 'x'),
              strcat('40xCPU off=6 speedup=', num2str(cpu40speedup_6), 'x'),
              strcat('10xCPU off=2 speedup=', num2str(cpu10speedup_2), 'x'),
              strcat('10xCPU off=6 speedup=', num2str(cpu10speedup_6), 'x'),
              strcat('single CPU off=2'),
              strcat('single CPU off=6')
          });
l.Location = 'northeast';
l.Box = 'off';
l.FontSize = 16;
title('Local Intensity Scale')
xlabel('Image size in GB');
ylabel('Processing time in seconds');
set(gcf, 'Position', [10, 10, 1300, 1100])
print('localIntensityScaleCpuVsGpu.eps' ,'-depsc','-painters','-loose','-cmyk');
print('localIntensityScaleCpuVsGpu.jpg' ,'-djpeg','-painters','-loose','-cmyk');


% -------------------------------------------------------------------------
% Recursive Filter plot
% -------------------------------------------------------------------------
xx=figure(2);
clf;
hold on;
format_figure(xx);

[~,gV100_B]=plotDataSimple(['../BenchmarkResults/v100/', 'BenchmarkBsplineTest.h5'], 0, 1);
[~,gP100_B]=plotDataSimple(['../BenchmarkResults/p100/', 'BenchmarkBsplineTest.h5'], 0, 1);
[~,g1080_B]=plotDataSimple(['../BenchmarkResults/Furiosa/', 'BenchmarkBsplineTest.h5'], 0, 1);
[~,gTITAN_B]=plotDataSimple(['../BenchmarkResults/ParticulatorOpenMP/', 'BenchmarkBsplineTest.h5'], 0, 1);
[~,g1080ti_B]=plotDataSimple(['../BenchmarkResults/1080ti/', 'BenchmarkBsplineTest.h5'], 0, 1);
[~,g1080ti2_B]=plotDataSimple(['../BenchmarkResults/1080ti2/', 'BenchmarkBsplineTest.h5'], 0, 1);

[cCPU40_B,~]=plotDataSimple(['../BenchmarkResults/Furiosa/', 'BenchmarkBsplineTest.h5'], 1, 0);
[cCPU10_B,~]=plotDataSimple(['../BenchmarkResults/ParticulatorOpenMP/', 'BenchmarkBsplineTest.h5'], 1, 0);
[cCPU_B,~]=plotDataSimple(['../BenchmarkResults/ParticulatorSingleCPU/', 'BenchmarkBsplineTest.h5'], 1, 0);

v100speedup_B=round(mean(cCPU_B ./ gV100_B));
p100speedup_B=round(mean(cCPU_B ./ gP100_B));
g1080speedup_B=round(mean(cCPU_B(1:15) ./ g1080_B));
gTITANspeedup_B=round(mean(cCPU_B ./ gTITAN_B));
g1080tispeedup_B=round(mean(cCPU_B ./ g1080ti_B));
g1080ti2speedup_B=round(mean(cCPU_B ./ g1080ti2_B));

cpu40speedup_B=round(mean(cCPU_B(1:15) ./ cCPU40_B));
cpu10speedup_B=round(mean(cCPU_B ./ cCPU10_B));

l = legend( { strcat('v100 speedup=', num2str(v100speedup_B), 'x'),
              strcat('p100 speedup=', num2str(p100speedup_B), 'x'),
              strcat('GTX 1080 speedup=', num2str(g1080speedup_B), 'x'),
              strcat('TITAN X speedup=', num2str(gTITANspeedup_B), 'x'),
              strcat('1080ti speedup=', num2str(g1080tispeedup_B), 'x'),
              strcat('1080ti2 speedup=', num2str(g1080ti2speedup_B), 'x'),
              strcat('40xCPU speedup=', num2str(cpu40speedup_B), 'x'),
              strcat('10xCPU speedup=', num2str(cpu10speedup_B), 'x'),
              strcat('single CPU'),
          });
l.Location = 'northeast';
l.Box = 'off';
l.FontSize = 16;      
title('Recursive Filter (k0=18)')
set(gcf, 'Position', [2500, 10, 1300, 1100])
axis([-Inf Inf -Inf 8]);   
xlabel('Image size in GB');
ylabel('Processing time in seconds');
print('recursiveCpuVsGpu.eps' ,'-depsc','-painters','-loose','-cmyk');
print('recursiveCpuVsGpu.jpg' ,'-djpeg','-painters','-loose','-cmyk');

% -------------------------------------------------------------------------
% Recursive Filter with varying K0 size (fixed img size) plot
% -------------------------------------------------------------------------
xx=figure(3);
clf;
hold on;
format_figure(xx);

[~,gTITAN_B]=plotDataSimple(['../BenchmarkResults/ParticulatorOpenMP/', 'BenchmarkBsplineVsK0sizeTest.h5'], 0, 1);
[~,gTITAN_B]=plotDataSimple(['../BenchmarkResults/ParticulatorOpenMP/', 'BenchmarkBsplineVsK0sizeTest8GB.h5'], 0, 1);
[cCPU10_B,~]=plotDataSimple(['../BenchmarkResults/ParticulatorOpenMP/', 'BenchmarkBsplineVsK0sizeTest.h5'], 1, 0);
[cCPU10_B,~]=plotDataSimple(['../BenchmarkResults/ParticulatorOpenMP/', 'BenchmarkBsplineVsK0sizeTest8GB.h5'], 1, 0);

l = legend( { 'TITAN-X GPU 4GB',
              'TITAN-X GPU 8GB',
              '10xCPU 4GB',
              '10xCPU 8GB',
          });
      
l.Location = 'east';
l.Box = 'off';
l.FontSize = 16;      
title('Recursive Filter with filter length k0=1-65')
set(gcf, 'Position', [2500, 10, 1300, 1100])
axis([-Inf Inf 0 5.5]);   
xlabel('k0 - size of filter');
ylabel('Processing time in seconds');
print('recursiveWithK0changesCpuVsGpu.eps' ,'-depsc','-painters','-loose','-cmyk');
print('recursiveWithK0changesCpuVsGpu.jpg' ,'-djpeg','-painters','-loose','-cmyk');

% -------------------------------------------------------------------------
% Utils
% -------------------------------------------------------------------------

function [cpuData, gpuData]=plotDataSimple(fileName, showCpu, showGpu)
    ad = load_analysis_data(fileName);
    x=ad.ticksValue;
    cpuData=[];
    gpuData=[];
    if showCpu == 1
        [cpuData, cpuErr]=getMeanMeasurements(ad.CpuTime, ad.numOfRepetitions, ad.numOfRepetitionsToSkip);
        errorbar(x, cpuData,cpuErr);
    end
    if showGpu == 1
        [gpuData, gpuErr]=getMeanMeasurements(ad.GpuDeviceTimeFull, ad.numOfRepetitions, ad.numOfRepetitionsToSkip);
        errorbar(x, gpuData,gpuErr);
    end
    
    ticksStep = 4;
    ticksVal = ad.ticksValue(1:ticksStep:length(ad.ticksValue));
    set(gca,'XTick', ticksVal)
    set(gca,'XTickLabel', num2str(ticksVal/ad.xNormalizer,strcat('%.',num2str(ad.numberOfDecimalPointsX),'f')));    
end


function plotData(fileName, colorShift)
    ad = load_analysis_data(fileName);
    
    [cpuData, cpuErr]=getMeanMeasurements(ad.CpuTime, ad.numOfRepetitions, ad.numOfRepetitionsToSkip);
    [gpuData, gpuErr]=getMeanMeasurements(ad.GpuDeviceTimeFull, ad.numOfRepetitions, ad.numOfRepetitionsToSkip);

    x=ad.ticksValue;
    
    cm_type = 'parula(5)';
    cm = colormap(cm_type)
    errorbar(x, cpuData,cpuErr, 'color', cm(3 + colorShift,:));
    errorbar(x, gpuData,gpuErr, 'color', cm(1 + colorShift,:));
    
%     axis([0 Inf 0 Inf]);    
    set(gca,'XTick', ad.ticksValue)
    set(gca,'XTickLabel', num2str(ad.ticksValue/ad.xNormalizer,strcat('%.',num2str(ad.numberOfDecimalPointsX),'f')));
    l = legend({'CPU', 'GPU'});
    l.Location = 'best';
    l.Box = 'off';
    l.FontSize = 20;
   
    xlabel(ad.xTitle');
    ylabel(ad.yTitle');            
    
    title(ad.plotTitle');    
end

function [out, maxErr]=getMeanMeasurements(data, noOfRep, skipSteps)
    numOfOutElements = length(data) / noOfRep;
    out = zeros(numOfOutElements,1);
    maxErr = zeros(numOfOutElements,1);
    i = 1;
    for idx=1:noOfRep:length(data)-(noOfRep-1)
        out(i) = mean(data(idx + skipSteps : idx + noOfRep - 1));
        maxErr(i) = std(data(idx + skipSteps:idx+noOfRep - 1));
        i = i + 1;
    end
end

function [out, maxErr]=getMinMeasurements(data, noOfRep, skipSteps)
    numOfOutElements = length(data) / noOfRep;
    out = zeros(numOfOutElements,1);
    maxErr = zeros(numOfOutElements,1);
    i = 1;
    for idx=1:noOfRep:length(data)-(noOfRep-1)
        out(i) = min(data(idx + skipSteps : idx + noOfRep - 1));
        maxErr(i) = std(data(idx + skipSteps:idx+noOfRep - 1));
        i = i + 1;
    end
end

end
