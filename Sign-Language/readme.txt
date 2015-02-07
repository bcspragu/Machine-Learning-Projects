Classify the time-series data as one of 95 different Auslan (Australian Sign Language) signs.

Samples: 6650
Dimensionality: 15
Target Labels: The meaning of each sign as a string (e.g. â€˜aliveâ€™, â€˜hurryâ€™) *scikit-learn handles non-numeric target labels like this without any modification

Feature Information:

x: 
- Continuous. 
- Description: x position between -1 and 1. Units are *approximately* meters. 
y: 
- Continuous. 
- Description: y position between -1 and 1. Units are approximately meters. 
z: 
- Continuous. 
- Description: z position between -1 and 1. Units are not meters. 
This space should not really be treated as linear, although it is safe to 
treat it as monotonically increasing. 
roll: 
- Continuous. 
- Description: roll with 0 meaning "palm down", rotating clockwise through to a maximum of 1 (not included), which is also "palm down". 
pitch: 
- Has a value of -1, indicating that it is not available for this data. 
Should be ignored. 
yaw: 
- Has a value of -1, indicating that it is not available for this data. 
Should be ignored. 
thumb: 
- Continuous. 
- Description: Thumb bend. has a value of 0 (straight) to 1 (fully bent). 
fore: 
- Continuous. 
- Description: Forefinger bend. has a value of 0 (straight) to 1 (fully bent). 
index: 
- Continuous. 
- Description: Index finger bend. has a value of 0 (straight) to 1 (fully bent). 
ring: 
- Continuous. 
- Description: Ring finger bend. has a value of 0 (straight) to 1 (fully bent). 
little: 
- In this case, it is a copy of ring bend. Should be ignored. 
keycode: 
- Indicates which key was pressed on the glove. Should be ignored. 
gs1: 
- glove state 1 Should be ignored. 
gs2: 
- glove state 2 should be ignored. 
Receiver values: 
- Determines if all receivers received values from all transmitters. A value of 0x3F indicates all receivers received information from all transmitters. Other values indicate this is not the case. 
