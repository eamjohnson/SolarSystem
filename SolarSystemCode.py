#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 09:13:30 2018

@author: elizabethjohnson
"""

from numpy import *
from scipy.integrate import ode
import matplotlib.pylab as plt
from matplotlib.pylab import *

'''
solving system of equation given by:
dr/dt=v
dv/dt=GMr^-2
'''

AU=1.496E11 #meters

#Make function to pass to integrator eqivalent to y(t) but it is a vector. 

def deriv(t,y):
    G=6.674E-11 # (m^3 kg^-1 s^-2)
    M=1.989E30 # mass of sun in kg
    M2=5.97E24 # earth
    M3 = 1.898E27 # jupiter
    M4=5.683E26 # saturn
    M5=1.024E26 # neptune
    M6=8.681E25 # uranus
    M7=6.39E23 # mars
    M8=4.867E24 # venus
    M9=1.989E30 #mercury 

    ### the input y has all this info:
    r1_x,r1_y,r2_x,r2_y,r3_x,r3_y,r4_x,r4_y,r5_x,r5_y,r6_x,r6_y,r7_x,r7_y,r8_x,r8_y,r9_x,r9_y,v1_x,v1_y,v2_x,v2_y,v3_x,v3_y,v4_x,v4_y,v5_x,v5_y,v6_x,v6_y,v7_x,v7_y,v8_x,v8_y,v9_x,v9_y=y 

    #####
    r12=sqrt((r1_x-r2_x)**2 + (r1_y-r2_y)**2)
    r23=sqrt((r2_x-r3_x)**2 + (r2_y-r3_y)**2)
    r13=sqrt((r1_x-r3_x)**2 + (r1_y-r3_y)**2)
    r14=sqrt((r1_x-r4_x)**2 + (r1_y-r4_y)**2)
    r24=sqrt((r2_x-r4_x)**2 + (r2_y-r4_y)**2)
    r34=sqrt((r3_x-r4_x)**2 + (r3_y-r4_y)**2)
    r15=sqrt((r1_x-r5_x)**2 + (r1_y-r5_y)**2)
    r25=sqrt((r2_x-r5_x)**2 + (r2_y-r5_y)**2)
    r35=sqrt((r3_x-r5_x)**2 + (r3_y-r5_y)**2)
    r45=sqrt((r4_x-r5_x)**2 + (r4_y-r5_y)**2)
    r16=sqrt((r1_x-r6_x)**2 + (r1_y-r6_y)**2)
    r26=sqrt((r2_x-r6_x)**2 + (r2_y-r6_y)**2)
    r36=sqrt((r3_x-r6_x)**2 + (r3_y-r6_y)**2)
    r46=sqrt((r4_x-r6_x)**2 + (r4_y-r6_y)**2)
    r56=sqrt((r5_x-r6_x)**2 + (r5_y-r6_y)**2)
    r17=sqrt((r1_x-r7_x)**2 + (r1_y-r7_y)**2)
    r27=sqrt((r2_x-r7_x)**2 + (r2_y-r7_y)**2)
    r37=sqrt((r3_x-r7_x)**2 + (r3_y-r7_y)**2)
    r47=sqrt((r4_x-r7_x)**2 + (r4_y-r7_y)**2)
    r57=sqrt((r5_x-r7_x)**2 + (r5_y-r7_y)**2)
    r67=sqrt((r6_x-r7_x)**2 + (r6_y-r7_y)**2)
    r18=sqrt((r1_x-r8_x)**2 + (r1_y-r8_y)**2)
    r28=sqrt((r2_x-r8_x)**2 + (r2_y-r8_y)**2)
    r38=sqrt((r3_x-r8_x)**2 + (r3_y-r8_y)**2)
    r48=sqrt((r4_x-r8_x)**2 + (r4_y-r8_y)**2)
    r58=sqrt((r5_x-r8_x)**2 + (r5_y-r8_y)**2)
    r68=sqrt((r6_x-r8_x)**2 + (r6_y-r8_y)**2)
    r78=sqrt((r7_x-r8_x)**2 + (r7_y-r8_y)**2)
    r19=sqrt((r1_x-r9_x)**2 + (r1_y-r9_y)**2)
    
    a1_x = (-G*M2*(r1_x - r2_x)/r12**3 - G*M3*(r1_x - r3_x)/r13**3 - G*M4*(r1_x - r4_x)/r14**3 - G*M5*(r1_x - r5_x)/r15**3 - G*M6*(r1_x - r6_x)/r16**3)
    
    a1_y = (-G*M2*(r1_y - r2_y)/r12**3 - G*M3*(r1_y - r3_y)/r13**3 - G*M4*(r1_y - r4_y)/r14**3 - G*M5*(r1_y - r5_y)/r15**3 - G*M6*(r1_y - r6_y)/r16**3)
    
    a2_x = (-G*M3*(r2_x - r3_x)/r23**3 - G*M*(r2_x - r1_x)/r12**3 - G*M4*(r2_x - r4_x)/r24**3 - G*M5*(r2_x - r5_x)/r25**3 - G*M6*(r2_x - r6_x)/r26**3 - G*M7*(r2_x - r7_x)/r27**3 - G*M8*(r2_x - r8_x)/r28**3)
    
    a2_y = (-G*M3*(r2_y - r3_y)/r23**3 - G*M*(r2_y - r1_y)/r12**3 - G*M4*(r2_y - r4_y)/r24**3 - G*M5*(r2_y - r5_y)/r25**3 - G*M6*(r2_y - r6_y)/r26**3 - G*M7*(r2_y - r7_y)/r27**3 - G*M8*(r2_y - r8_y)/r28**3)
    
    a3_x = (-G*M*(r3_x - r1_x)/r13**3 - G*M2*(r3_x - r2_x)/r23**3 - G*M4*(r3_x - r4_x)/r34**3 - G*M5*(r3_x - r5_x)/r35**3 - G*M6*(r3_x - r6_x)/r36**3)
    
    a3_y = (-G*M*(r3_y - r1_y)/r13**3 - G*M2*(r3_y - r2_y)/r23**3 - G*M4*(r3_y - r4_y)/r34**3 - G*M5*(r3_y - r5_y)/r35**3 - G*M6*(r3_y - r3_y)/r36**3)
    
    a4_x = (-G*M*(r4_x - r1_x)/r14**3 - G*M2*(r4_x - r2_x)/r24**3 - G*M3*(r4_x - r3_x)/r34**3 - G*M5*(r4_x - r5_x)/r45**3 - G*M6*(r4_x - r6_x)/r46**3)
    
    a4_y = (-G*M*(r4_y - r1_y)/r14**3 - G*M2*(r4_y - r2_y)/r24**3 - G*M3*(r4_y - r3_y)/r34**3 - G*M5*(r4_y - r5_y)/r45**3 - G*M6*(r4_y - r4_y)/r46**3)
    
    a5_x = (-G*M*(r5_x - r1_x)/r15**3 - G*M2*(r5_x - r2_x)/r25**3 - G*M3*(r5_x - r3_x)/r35**3 - G*M4*(r5_x - r4_x)/r45**3 - G*M6*(r5_x - r6_x)/r56**3)
    
    a5_y = (-G*M*(r5_y - r1_y)/r15**3 - G*M2*(r5_y - r2_y)/r25**3 - G*M3*(r5_y - r3_y)/r35**3 - G*M4*(r5_y - r4_y)/r45**3 - G*M6*(r5_y - r6_y)/r56**3)
   
    a6_x = (-G*M*(r6_x - r1_x)/r16**3 - G*M2*(r6_x - r2_x)/r26**3 - G*M3*(r6_x - r3_x)/r36**3 - G*M4*(r6_x - r4_x)/r46**3 - G*M5*(r6_x - r5_x)/r56**3)
    
    a6_y = (-G*M*(r6_y - r1_y)/r16**3 - G*M2*(r6_y - r2_y)/r26**3 - G*M3*(r6_y - r3_y)/r36**3 - G*M4*(r6_y - r4_y)/r46**3 - G*M5*(r6_y - r5_y)/r56**3)
    
    a7_x = (-G*M*(r7_x - r1_x)/r17**3 - G*M2*(r7_x - r2_x)/r27**3 - G*M3*(r7_x - r3_x)/r37**3 - G*M4*(r7_x - r4_x)/r47**3 - G*M5*(r7_x - r5_x)/r57**3 - G*M6*(r7_x - r6_x)/r67**3 - G*M8*(r7_x - r8_x)/r78**3)
    
    a7_y = (-G*M*(r7_y - r1_y)/r17**3 - G*M2*(r7_y - r2_y)/r27**3 - G*M3*(r7_y - r3_y)/r37**3 - G*M4*(r7_y - r4_y)/r47**3 - G*M5*(r7_y - r5_y)/r57**3 - G*M6*(r7_y - r6_y)/r67**3 - G*M8*(r7_y - r8_y)/r78**3)
    
    a8_x = (-G*M*(r8_x - r1_x)/r18**3 - G*M2*(r8_x - r2_x)/r28**3 - G*M3*(r8_x - r3_x)/r38**3 - G*M4*(r8_x - r4_x)/r48**3 - G*M5*(r8_x - r5_x)/r58**3 - G*M6*(r8_x - r6_x)/r68**3 - G*M7*(r8_x - r7_x)/r78**3)
    
    a8_y = (-G*M*(r8_y - r1_y)/r18**3 - G*M2*(r8_y - r2_y)/r28**3 - G*M3*(r8_y - r3_y)/r38**3 - G*M4*(r8_y - r4_y)/r48**3 - G*M5*(r8_y - r5_y)/r58**3 - G*M6*(r8_y - r6_y)/r68**3 - G*M7*(r8_y - r7_y)/r78**3)
    
    a9_x = (-G*M*(r9_x - r1_x)/r19**3) ## realistically, Mercury isn't really going to affect the other planets too much so I left out their effects 
            
    a9_y = (-G*M*(r9_y - r1_y)/r19**3)       
    
    return [v1_x,v1_y,v2_x,v2_y,v3_x,v3_y,v4_x,v4_y,v5_x,v5_y,v6_x,v6_y,v7_x,v7_y,v8_x,v8_y,v9_x,v9_y,a1_x,a1_y,a2_x,a2_y,a3_x,a3_y,a4_x,a4_y,a5_x,a5_y,a6_x,a6_y,a7_x,a7_y,a8_x,a8_y,a9_x,a9_y]
    #return [v_r,v_phi,a_r,a_phi] #returns the time derivative dy(t)/dt
'''
For this example I used the perihelion distance of Earth and tried to 
pick the correct angular and radial velocities. I am a little off
because in the second plot the peak should be where the two line cross
because that would be the correct aphelion. I couldn't find a 
good value for the velocities so I took educated guesses. 
N.B. 

If v_r0=0 is used you will get a circular orbit.

'''

N=200.001 #number of Earth years to integrate
D=.9832899 #Distance at r0 (in AU)
D_J = 5.2  ## jupiter
D_S = 9.6 ## saturn
D_N = 30.1 ## neptune
D_U = 18.4 ## uranus
D_M = 1.524 #mars
D_V = 0.723 ## venus
D_mer = 0.39 ## mercury


#intital conditions 
r0 = [0,0.0000001,D*AU,0.0000001,D_J*AU,0.0000001,D_S*AU,0.0000001,D_N*AU,0,D_U*AU,0,D_M*AU,0,D_V*AU,0,D_mer*AU,0,0,0,0,30000,0,13000,0,9680,0,5430,0,6800,0,24080,0,35020,0,47400]                  # r1_x,r1_y, etc.
t0=0 #intial time = 0


'''
This is initializing the integrator class:
ode(deriv) is saying make an integrator that integrates the function deriv

.set_integrator('dopri5') sets the method to Runge-Kutta4 

e.g. 
.set_integrator('dop853') gives RK8

see :
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
 
'''
#************************************#

r=ode(deriv).set_integrator('dopri5')

#***********************************#


r.set_initial_value(r0,t0) #self-explanatory

# note that in deriv y is a 4-d vecotr so our initial value (r0) is als0
# a 4-d vector

# empty arrays for results

r1_x_vals=empty(0)
r1_y_vals=empty(0)
r2_x_vals=empty(0)
r2_y_vals=empty(0)
r3_x_vals=empty(0)
r3_y_vals=empty(0)
r4_x_vals=empty(0)
r4_y_vals=empty(0) 
r5_x_vals=empty(0)
r5_y_vals=empty(0)
r6_x_vals=empty(0)
r6_y_vals=empty(0) 
r7_x_vals=empty(0)
r7_y_vals=empty(0) 
r8_x_vals=empty(0)
r8_y_vals=empty(0)
r9_x_vals=empty(0)
r9_y_vals=empty(0) 


Jupiter_energy=empty(0)

# Set the time array for integration. Defined N above so it is easily stated
# in terms of Earth years. 
t=linspace(0,N*24*365*60*60,2000)
timeperenergy = []

#This is where the integration actually happens
n=0
for dt in t:
    n+= 1
    integr = r.integrate(dt)
    r1_x_vals=append(r1_x_vals,integr[0]/AU)
    r1_y_vals=append(r1_y_vals,integr[1]/AU)
    r2_x_vals=append(r2_x_vals,integr[2]/AU)
    r2_y_vals=append(r2_y_vals,integr[3]/AU)
    r3_x_vals=append(r3_x_vals,integr[4]/AU)
    r3_y_vals=append(r3_y_vals,integr[5]/AU) 
    r4_x_vals=append(r4_x_vals,integr[6]/AU)
    r4_y_vals=append(r4_y_vals,integr[7]/AU)
    r5_x_vals=append(r5_x_vals,integr[8]/AU)
    r5_y_vals=append(r5_y_vals,integr[9]/AU)
    r6_x_vals=append(r6_x_vals,integr[10]/AU)
    r6_y_vals=append(r6_y_vals,integr[11]/AU)
    r7_x_vals=append(r7_x_vals,integr[12]/AU)
    r7_y_vals=append(r7_y_vals,integr[13]/AU)
    r8_x_vals=append(r8_x_vals,integr[14]/AU)
    r8_y_vals=append(r8_y_vals,integr[15]/AU)
    r9_x_vals=append(r9_x_vals,integr[16]/AU)
    r9_y_vals=append(r9_y_vals,integr[17]/AU)
    if n % 100 == 0: 
        Jupiter_energy=append(Jupiter_energy,0.5*1.898E27*(sqrt(integr[22]**2 + integr[23]**2))**2+1.898E27*1.989E30*6.674E-11/(sqrt(integr[4]**2 + integr[5]**2)))
        timeperenergy.append(dt)
        

#Plot the orbit in a cartesian space 
fig = plt.figure()
title('Solar System')
plot(r1_x_vals, r1_y_vals, 'r.', label='Sun')
plot(r9_x_vals, r9_y_vals, '.', color='gold', alpha=0.25, label='Mercury')
plot(r8_x_vals, r8_y_vals, '.', color='orchid', alpha=0.25, label='Venus')
plot(r2_x_vals, r2_y_vals, 'b.', alpha=0.25, label='Earth')
plot(r7_x_vals, r7_y_vals, '.', color='coral', alpha=0.25, label='Mars')
plot(r3_x_vals, r3_y_vals, 'm.', alpha=0.25, label='Jupiter')
plot(r4_x_vals, r4_y_vals, '.', color='orange', alpha=0.25, label='Saturn')
plot(r6_x_vals, r6_y_vals, '.', color='lawngreen', alpha=0.25, label='Uranus')
plot(r5_x_vals, r5_y_vals, '.', color='skyblue', alpha=0.25, label='Neptune')
legend()
ylabel('Distance (AU)')
xlabel('Distance (AU)')
show()


