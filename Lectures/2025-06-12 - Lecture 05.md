
habe die ersten 30min verpasst...

![[Pasted image 20250612112605.png]]

die rote linie ist am besten, weil es nicht zu schnell zu genau wird 
aber 0.0001 und 0.001 sind meistens der sweat spot

![[Pasted image 20250612112643.png]]

hier wollen wir, dass wir overshooting vermeiden, lieber treffen wir das ziel nicht genau als daran vorbei zu schießen. ein auto, dass in eine parklücke fahren soll soll lieber früher stehenbleiben als zu spät und dann in ein anderes Auto zu krachen.

![[Pasted image 20250612112933.png]]

Das problem ist, dass man beim overfitting, wenn neue daten (z.b. spam mails) dazukommen diese nicht mehr ordentlich klassifiziert werden können oder so.

man muss  auch aufpassen, dass man bei test datensätzen nicht overfitted, wenn, wenn man sozusagen nur für das test datenset optimiert ist es eher schlecht, dann muss man sich aus der echten welt ein neues datenset holen...


### Random Split
![[Pasted image 20250612113155.png]]
we do not do randomslip on users because new user will not fit in usually.
#### participation wise split
so we do random slip within the user.

training has the most data because it help back propagate (why is this important?)

then for validation we take another batch that is double the size of the expected users we use in the real world

train 70
validate 20
test in real world 10

![[Pasted image 20250612113645.png]]
so wen can also train on user 1 and 2 and validate on user 2 and 3


Cross-Validation ▪ Leave-k-Out Cross-Validation ▪ Leave-One-Group-Out Cross-Validation ▪ Leave-One-Participant-Out Cross-Validation

we can get multible accuary values if the split the 
the lower the standart deviation the better the result.

what is the best way to do it?

## Confusion Matrix

![[Pasted image 20250612114224.png]]
this is a binary dataset in a confusion matrix
we want the most values inside the diagonal with true positives or true negatives.

but what is a confusion matrix?


with a multi class classification confusion matrix we also want the diagonal



a loss function can be all kinds of things but a .Precision, Recall, and F1 score ▪ Confusion Matrix Ev is way better
explani this pleease