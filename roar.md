SELECT

FROM

ORDER

BY

WHERE

OR

AND

IN

BETWEEN

YEAR

SUM

```
SELECT orders.orderNumber, ROUND(SUM(quantityOrdered * priceEach),2)AS tot
FROM orderdetails, orders, customers
WHERE	
	orderdetails.orderNumber=orders.orderNumber
    AND orders.customerNumber = customers.customerNumber
GROUP BY orders.orderNumber
ORDER BY customerNumber```
