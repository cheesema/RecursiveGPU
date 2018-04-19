function RecursiveBenchmark()
addpath('/Users/gonciarz/Documents/MOSAIC/work/repo/RecursiveGPU/APRBench/Matlab');

% name='asdf.h5';
analysis_root='/Users/gonciarz/Documents/MOSAIC/work/repo/RecursiveGPU/build/';

xx=figure(1);
clf;
hold on;
format_figure(xx);
plotData([analysis_root, 'BenchmarkLocalIntensityScaleTestOffset2.h5'], 1);
plotData([analysis_root, 'BenchmarkLocalIntensityScaleTestOffset6.h5'], 1);
% plotData([analysis_root, 'asdfFalcon.h5'], 1);


function plotData(fileName, plotNum)
    ad = load_analysis_data(fileName);
    ad
%     ad.numOfRepetitions
%     ad.GpuDeviceTimeYdir
%     ad.GpuDeviceTimeXdir
%     ad.GpuDeviceTimeZdir
    
    % Test options
    numOfRep = ad.numOfRepetitions;
    skipNumOfFirstElements=ad.numOfRepetitionsToSkip;
    
    [cpuData, cpuErr]=getMeanMeasurements(ad.CpuTime, numOfRep, skipNumOfFirstElements);
    [gpuData, gpuErr]=getMeanMeasurements(ad.GpuDeviceTimeFull, numOfRep, skipNumOfFirstElements);

    x=ad.ticksValue;
    size(cpuData)
    size(gpuData)
    cpuData./gpuData
    
    figure(plotNum);
%     format_figure(gcf);
    hold on;
    cm_type = 'parula(5)';
    cm = colormap(cm_type)
    errorbar(x, cpuData,cpuErr, 'color', cm(3,:));
    errorbar(x, gpuData,gpuErr, 'color', cm(1,:));
    
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
%     print('recursiveCpuVsGpu.eps' ,'-depsc','-painters','-loose','-cmyk');
    
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
