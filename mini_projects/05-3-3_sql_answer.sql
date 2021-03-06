/* Welcome to the SQL mini project. */

/* Q1: Some of the facilities charge a fee to members, but some do not.
Please list the names of the facilities that do. */

SELECT name as fac_membercost
FROM Facilities
WHERE membercost > 0;

/* Q2: How many facilities do not charge a fee to members? */

SELECT COUNT(*)
FROM Facilities
WHERE membercost = 0;

/* Q3: How can you produce a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost?
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */

SELECT facid,
       name,
       membercost,
       monthlymaintenance
FROM Facilities
WHERE (membercost / monthlymaintenance) < 0.2;

/* Q4: How can you retrieve the details of facilities with ID 1 and 5?
Write the query without using the OR operator. */

SELECT *
FROM Facilities
WHERE facid IN (1, 5);

/* Q5: How can you produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100? Return the name and monthly maintenance of the facilities
in question. */

SELECT name,
	     monthlymaintenance,
       CASE WHEN monthlymaintenance < 100 THEN 'cheap'
            ELSE 'expensive'
       END AS cheap_or_expensive
FROM Facilities;

/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Do not use the LIMIT clause for your solution. */

SELECT *
FROM Members
WHERE joindate = (SELECT MAX(joindate) FROM Members);

/* Q7: How can you produce a list of all members who have used a tennis court?
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */

SELECT DISTINCT CONCAT(mem.firstname, ' ', mem.surname) as member,
	     			    fac.name
FROM Facilities as fac
INNER JOIN Bookings as book
    ON fac.facid = book.facid
INNER JOIN Members as mem
    ON book.memid = mem.memid
WHERE fac.facid IN (0, 1)
AND mem.memid != 0
ORDER BY member;

/* Q8: How can you produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30? Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */

SELECT  Facilities.name,
        CONCAT(Members.firstname, ' ', Members.surname) as user,
        CASE WHEN Bookings.memid = 0 THEN Bookings.slots * Facilities.guestcost
             ELSE Bookings.slots * Facilities.membercost
             END AS cost_per_booking
FROM Bookings
LEFT JOIN Facilities ON Bookings.facid = Facilities.facid
LEFT JOIN Members ON Bookings.memid = Members.memid
WHERE LEFT(starttime, 10) = '2012-09-14'
AND CASE WHEN Bookings.memid = 0 THEN Bookings.slots * Facilities.guestcost
         ELSE Bookings.slots * Facilities.membercost
         END > 30
ORDER BY cost_per_booking DESC;

/* Q9: This time, produce the same result as in Q8, but using a subquery. */

SELECT *
FROM
(SELECT Facilities.name,
        CONCAT(Members.firstname, ' ', Members.surname) as user,
        CASE WHEN Bookings.memid = 0 THEN Bookings.slots * Facilities.guestcost
             ELSE Bookings.slots * Facilities.membercost
             END AS cost_per_booking
FROM Bookings
LEFT JOIN Facilities ON Bookings.facid = Facilities.facid
LEFT JOIN Members ON Bookings.memid = Members.memid
WHERE LEFT(starttime, 10) = '2012-09-14'
) as sub_query
WHERE cost_per_booking > 30
ORDER BY cost_per_booking DESC;

/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */

SELECT  Facilities.name,
        SUM(CASE WHEN Bookings.memid = 0 THEN Bookings.slots * Facilities.guestcost
             ELSE Bookings.slots * Facilities.membercost
             END ) AS total_rev
FROM Bookings
LEFT JOIN Facilities ON Bookings.facid = Facilities.facid
GROUP BY Facilities.name
HAVING total_rev < 1000
ORDER BY total_rev DESC;
