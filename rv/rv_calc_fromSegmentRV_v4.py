# 2023-10-20.  
# author: M. Kuzuhara 

import sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import glob
from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import median_filter
from scipy import special, stats

##########


show_fig = False
save_removed_segorder_IDs = False


def f(x, b):

    f = b + 0*x

    return f

## read files
def file_read(target, sumarized_regular_rv = "../median_RV_werr_202308_v2.dat"):

    datas = glob.glob("fitresult_*dat")
    datas = np.sort(datas)
    rvonly = np.loadtxt(sumarized_regular_rv, dtype=str)
    rvonly = rvonly[rvonly[:,4] == target]
    rverronly = np.array(rvonly[:,2], dtype=float)
    rvtotal = np.array(rvonly[:,1], dtype=float)
    #offset = np.array(rvonly[:,5], dtype=float)
    #decimalyear = np.array(rvonly[:,0], dtype=float)
    telluric = np.array(rvonly[:,8], dtype=float)
    JD = np.array(rvonly[:,7], dtype=float)
    fname = np.array(rvonly[:,3])

    return datas, rvonly, rverronly, rvtotal, telluric, JD, fname  


def main(target_name, original_file_summary, c_bad_seg_1 = 5, c_bad_seg2 = 6, c_bad_epoch_inSeg = 2., c_scatter = 95, c_high_scatter_epoch = 4., \
           c_deviated_to_Terr_epoch = 6., c_deviated_epoch = 2., c_bad_chisq_sigma = 30.):
    
    datas, rvonly, rverronly, rvtotal, telluric, JD, fname  = file_read(target_name, original_file_summary)
    
    # summarize all fitresult arrays, telluric velocity, JDs into a single array
    # also make an array to index masked segments

    for i, row in enumerate(datas):

        data = np.loadtxt(row)
        if float(row.replace("fitresult_", "").replace(".dat", "")) != float(fname[i]):
            print("incorrect file number", row, fname[i]) 
            sys.exit(1)

        tellones = telluric[i] * np.ones(len(data))/1000.
        tellones = np.reshape(tellones, (len(data), 1))
        data = np.hstack( (data, tellones) )

        JDones = JD[i] * np.ones(len(data))
        JDones = np.reshape(JDones, (len(data), 1))
        data = np.hstack( (data, JDones) )

        dummy = np.zeros(len(data))
        dummy = np.reshape(dummy, (len(data), 1))
        data = np.hstack( (data, dummy) )

        if i == 0:
            
            alldata = np.copy(data)

        else:

            alldata = np.vstack((alldata, data)) 


    alldata = np.array(alldata)

    order = np.unique(alldata[:,17])
    seg = np.unique(alldata[:,18])

    scatters = np.ones(len(alldata))
    msk = np.ones(len(alldata))

    ## compute scatter of each segment
    for o in order:

        for s in seg: 

            same_seg = np.where((alldata[:,17] == o) & (alldata[:,18] == s) )
            
            if len(same_seg[0]) == 0:  continue 
            
            rvseg = alldata[same_seg][:,1]
            tellv = alldata[same_seg][:,19]
            rvtmp = rvseg + tellv
            mad = median_abs_deviation(rvtmp, scale="normal")
            stddev = np.nanstd(rvtmp, ddof=1)
            
            # mask segments with standard deviations much higher than median absolute deviation
            # such sgements are abnormal  
            if (mad == mad and stddev == stddev) and (stddev > c_bad_seg_1*mad):

                msk[same_seg] = 0

            # mask segments with median absolute deviation much higher than photon-noise errors
            # such sgements are abnormal
            if (mad == mad ) and (mad/np.nanmedian(alldata[:,2][same_seg]) > c_bad_seg2):

                msk[same_seg] = 0   

            elif mad != mad:

                msk[same_seg] = 0    

            ## mask rvs 2-sigma higher than mdedian absolute deviation in each segment 
            badindx = np.where(np.abs(rvtmp - np.nanmedian(rvtmp)) > c_bad_epoch_inSeg*mad )        
            badindx = same_seg[0][badindx]
            msk[badindx] = 0
        
            scatters[same_seg] = mad # save rv scatters at each segment
        
    ## mask RVs 
    msk = np.where(msk == 0)
    alldata[:,1][msk] = np.nan

    rvout = []
    
    ## masks wave ranges related to significant telluric absorptions
    wbad = np.where(alldata[:,0] < 1051.5)
    alldata[:,1][wbad] = np.nan
    wbad = np.where(  (alldata[:,0] > 1113) & (alldata[:,0] < 1160) )
    alldata[:,1][wbad] = np.nan
    wbad = np.where(  (alldata[:,0] > 1330) & (alldata[:,0] < 1497) )
    alldata[:,1][wbad] = np.nan

    # compute RVs at each pointing
    seg_order_IDs = []
    for i, row in enumerate(JD):

        indx = np.where(alldata[:,20] == JD[i] )
        data = alldata[indx]
        chisq = data[:,15]
        dof = data[:,16]
        order_ID = np.reshape(data[:,17], (len(data), 1))
        seg_ID = np.reshape(data[:,18], (len(data), 1))
        JD_tmp = np.ones(len(data))*JD[i]
        JD_tmp = np.reshape(JD_tmp, (len(data),1))
        seg_order_ID = np.hstack((JD_tmp, order_ID, seg_ID))

        rv = data[:,1] + data[:,19]
        internal_err_med = np.nanmedian(data[:,2])
        rverr = np.sqrt(scatters[indx]**2 + data[:,2]**2)
        # rv errors at each segment = square-root of segment-rv scatters and rv errors 

        # mask segments with highest 5% sctters   ####
        scatter_criterion = c_scatter
        #print("high scatter", np.percentile(scatters[indx], [scatter_criterion]))
        high_scatter_indx = np.where(scatters[indx] > np.percentile(scatters[indx], [scatter_criterion])[0])
        rv[high_scatter_indx] = np.nan 
        ##############################################

        # re-mask segments with scatters much higher than photon-noise errors  ####
        high_scatter_to_photon = np.where(scatters[indx] > c_high_scatter_epoch*data[:,2])
        rv[high_scatter_to_photon] = np.nan   
        #

        # mask segment that are significantly (> 6 sigma) deviated from median  
        rvscale = np.copy( (rv - np.nanmedian(rv))/rverr )  
        deviated = np.where(np.abs(rvscale) > c_deviated_to_Terr_epoch)
        rv[deviated] = np.nan 
        #  
    
        # mask segements that are highly deviated by 2 x median absolute deviation
        # of the segments
        mad_i = median_abs_deviation(rv[rv==rv], scale="normal")
        deviated = np.where( np.abs(rv - np.nanmedian(rv)) > c_deviated_epoch*mad_i)
        rv[deviated] = np.nan 
        #  

        ## Deviated points with 30 sigma bad chi-squares are removed
        gauss_sig = special.erfcinv(stats.chi2.sf(chisq, dof))*2**0.5
        #plt.plot(gauss_sig,rvscale,"o")
        #plt.show()
        badindx = np.where( (gauss_sig > c_bad_chisq_sigma) & (np.abs(rvscale) > c_deviated_to_Terr_epoch*3/2.) )
        rv[badindx] = np.nan

        # pick up only unmasked rvs
        good = np.where(rv == rv)

        if i == 0:

            seg_order_IDs = np.copy(seg_order_ID)

        else:

            seg_order_IDs = np.vstack((seg_order_IDs, seg_order_ID))    



        ## fit f(x)  = b to calculate RVs and their errors
        try:

            fit_rst0 = curve_fit(f, data[:,0][good], rv[good], p0 = (np.nanmedian(rv)), sigma=rverr[good])
            rvout.append([JD[i], fit_rst0[0][0]*1000, np.sqrt(fit_rst0[1][0][0])*1000, np.median(rverr)*1000, \
                          internal_err_med*1000, len(rv[good])])
                        
        except:

            print("pass")


        if show_fig is True:
            print("Frame ID, Julidan day and barycentric correction velocity", fname[i], JD[i], telluric[i])
            plt.xlim(1050,1750)
            plt.ylim(-1,1)
            plot_jd = data[:,0][good]
            plot_rv = rv[good] - np.median(rv[good])
            plot_err = rverr[good]
            good_precisions = np.where(plot_err < np.percentile(plot_err, [50])[0])
            best_fit = fit_rst0[0][0] - np.median(rv[good])
            plt.plot([1050, 1750], [best_fit, best_fit])
            plt.errorbar(plot_jd[good_precisions], plot_rv[good_precisions], yerr = plot_err[good_precisions], fmt="o", alpha=0.2)
            plt.show()

    rvout = np.array(rvout)
    rvout[:,1] = rvout[:,1] - np.median(rvout[:,1])
    seg_order_IDs = np.array(seg_order_IDs)
    
    np.savetxt('rvout_' + target_name + '.dat', rvout)

    if save_removed_segorder_IDs is True:

        np.savetxt("rmov_order_seg" + target_name +".dat", seg_order_IDs)

    return rvout

if __name__ == "__main__":

    target_name = sys.argv[1]
    original_rv_summary = sys.argv[2]
    main(target_name=target_name, original_file_summary = original_rv_summary)
