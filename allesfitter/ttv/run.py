#!/usr/bin/env python
import allesfitter

#fig = allesfitter.prepare_ttv_fit('.')
fig = allesfitter.show_initial_guess('.')

allesfitter.ns_fit('.')
allesfitter.ns_output('.')

#allesfitter.mcmc_fit('.')
#allesfitter.mcmc_output('.')
