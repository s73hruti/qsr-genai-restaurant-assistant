"""Hand-written (synthetic) knowledge documents: FAQs, policies, manager SOPs, training.

Each entry: (doc_id, title, [(heading, body), ...], [(question, expected_fact), ...]).
The questions are written with deliberately different vocabulary from the documents so that
retrieval has to do more than keyword matching. `expected_fact` must appear verbatim
(case-insensitive) in the document; a test enforces this.
"""
from __future__ import annotations

FAQS = [
    ("faq-refunds", "Refunds and order problems", [
        ("Wrong or missing items", "If your order is wrong or missing an item, open the order in the app and tap \"Report a problem\". We reply within 1 hour during opening times."),
        ("When refunds arrive", "Refunds for app payments are returned to the original payment method within 3 to 5 business days."),
    ], [("How many days until my money comes back after a refund?", "3 to 5 business days"),
        ("My bag was missing a drink, what do I do?", "Report a problem")]),
    ("faq-loyalty-tiers", "Loyalty program tiers", [
        ("Earning points", "Rewards members earn 1 point per dollar spent. Silver is the starting tier."),
        ("Tier thresholds", "Gold starts at 500 points in a calendar year and Platinum at 1500 points. Gold and Platinum members get early access to limited-time offers and double points on weekends."),
    ], [("How many points do I need to become a Gold member?", "500 points"),
        ("What extra perks come with Platinum status?", "double points on weekends")]),
    ("faq-points-redemption", "Redeeming loyalty points", [
        ("Redemption value", "Every 100 points can be exchanged for a 1 dollar reward at checkout. Rewards are applied automatically when you tap \"Use points\"."),
        ("Expiry", "Points expire 12 months after the date they were earned. The app shows your next expiry date on the rewards page."),
    ], [("After how long do my unused rewards points disappear?", "12 months after"),
        ("What is the cash value of one hundred points?", "1 dollar reward")]),
    ("faq-allergens-general", "Allergens and dietary needs", [
        ("Where to find allergen details", "Allergen details are listed on every menu item page. Our kitchens handle gluten, milk, egg and fish so we cannot guarantee any item is completely allergen free."),
        ("Shared equipment", "Vegetarian items are cooked on shared equipment with meat products. No menu item is certified vegan or halal."),
    ], [("Is the food cooked separately for vegetarians?", "shared equipment"),
        ("Can you promise a meal with zero allergen risk?", "cannot guarantee")]),
    ("faq-order-tracking", "Order tracking and pickup times", [
        ("Tracking", "After ordering in the app you can track preparation in real time."),
        ("Typical pickup time", "Drive-thru and curbside pickup orders are normally ready within 4 to 8 minutes, longer at peak lunch and dinner times."),
    ], [("Roughly how quickly will my curbside order be ready?", "4 to 8 minutes")]),
    ("faq-delivery", "Delivery", [
        ("Hours and fees", "Delivery is available from 8am to 10pm. Delivery fees depend on distance and are shown at checkout."),
        ("Free delivery", "Delivery is free on orders above 30 dollars when the address is within 3 km of the store."),
    ], [("What basket value qualifies for no delivery charge?", "above 30 dollars"),
        ("Until what time can I get food brought to my home?", "10pm")]),
    ("faq-payment", "Payment methods", [
        ("Accepted methods", "The app accepts debit and credit cards, mobile wallets and gift cards. Cash is accepted in store only."),
        ("Offers at checkout", "Eligible offers are applied automatically at checkout. Only one offer can be used per order."),
    ], [("Can I pay with cash when ordering through the app?", "in store only")]),
    ("faq-gift-cards", "Gift cards", [
        ("Amounts and validity", "Gift cards can be bought in amounts from 10 to 100 dollars and stay valid for 24 months from purchase."),
        ("Lost cards", "A lost physical card can be blocked in the app if it was registered to your account."),
    ], [("For how long can a present card be used?", "24 months")]),
    ("faq-receipts", "Receipts and invoices", [
        ("Digital receipts", "A tax invoice is emailed within 24 hours of every app order. You can also download it from the order history page."),
    ], [("How soon will I get my tax invoice by email?", "within 24 hours")]),
    ("faq-lost-items", "Lost property", [
        ("Items left in store", "Items left behind are kept for 30 days at the store where they were found. Call the store with a description to arrange collection."),
    ], [("I forgot my umbrella at a restaurant, how long will they keep it?", "30 days")]),
    ("faq-feedback", "Feedback and surveys", [
        ("Survey code", "Enter the survey code printed on your receipt in the app to share feedback and receive a free Soft Serve Cone on your next visit."),
        ("Response time", "Our customer care team replies to feedback within 2 business days."),
    ], [("What reward do I get for completing the receipt survey?", "free Soft Serve Cone")]),
    ("faq-catering", "Catering and large orders", [
        ("Notice period", "Orders of 20 or more burgers need at least 24 hours notice. Place large orders through the catering form in the app or by calling the store."),
    ], [("How much advance warning for a party order of twenty burgers?", "24 hours notice")]),
    ("faq-kids", "Kids menu favourites", [
        ("Popular with children", "The most popular items for children are the Classic Burger, Chicken Nuggets 6pc and the Soft Serve Cone. Ask for apple slices instead of fries at no extra charge."),
    ], [("Which items do children usually pick?", "Classic Burger, Chicken Nuggets 6pc")]),
    ("faq-app-login", "Signing in and password reset", [
        ("Resetting your password", "Tap \"Forgot password\" on the sign-in screen. The reset link sent to your email is valid for 30 minutes."),
        ("Locked accounts", "After 5 failed sign-in attempts the account is locked for 15 minutes."),
    ], [("The link to choose a new password stopped working, how long does it last?", "30 minutes"),
        ("How long am I locked out after typing the wrong password repeatedly?", "locked for 15 minutes")]),
    ("faq-app-problems", "App crashes and technical issues", [
        ("Troubleshooting", "Update to the latest app version, restart the phone and clear the app cache. If the problem continues use Help and then Contact support inside the app so we can attach diagnostic logs."),
    ], [("The application keeps freezing, what should I try first?", "Update to the latest app version")]),
    ("faq-delete-account", "Deleting your account and data", [
        ("Account deletion", "Request deletion under Settings, then Privacy. Your personal data is erased within 30 days. Order records needed for tax are kept in anonymised form."),
    ], [("How long does it take to wipe my personal information after I ask?", "within 30 days")]),
    ("faq-order-changes", "Changing or cancelling an order", [
        ("Edit window", "Orders can be edited within 2 minutes of placing them. Cancellation is free until the kitchen starts preparing the order."),
    ], [("Until when can I still modify what I just ordered?", "within 2 minutes")]),
    ("faq-curbside", "Curbside pickup", [
        ("How it works", "Park in a marked curbside bay and tap \"I'm here\" in the app. A team member brings your order to the car, usually within 3 minutes."),
    ], [("After I tell the app I've arrived outside, how long till someone brings the food?", "within 3 minutes")]),
    ("faq-accessibility", "Accessibility", [
        ("In store", "All stores have step-free entrances and accessible tables. Large print menus are available on request and the app supports screen readers."),
    ], [("Do you provide menus with bigger lettering?", "Large print menus")]),
    ("faq-opening-hours", "General opening hours", [
        ("Typical hours", "Most stores open between 6am and 7am and close between 10pm and 11pm. Three stores are open 24 hours. See each store page for exact hours."),
    ], [("How many restaurants never close?", "Three stores are open 24 hours")]),
    ("faq-offer-stacking", "Combining offers", [
        ("One offer per order", "Only one offer can be applied to each order. If several offers are eligible the app picks the one with the largest saving."),
    ], [("Can I use two discounts on the same purchase?", "one offer can be applied")]),
    ("faq-birthday", "Birthday reward", [
        ("Free treat", "Members receive a free Soft Serve Cone during their birthday week. The reward appears in the app 7 days before your birthday."),
    ], [("What do members get during the week of their birthday?", "free Soft Serve Cone")]),
    ("faq-student-discount", "Student discount", [
        ("Eligibility", "Show a valid student ID at participating stores to get 10 percent off. The discount cannot be combined with other offers and is not available for delivery."),
    ], [("Do pupils with a school card get a price cut?", "10 percent off")]),
    ("faq-nutrition", "Nutrition information", [
        ("Where to look", "Calories and allergens are shown on each menu item page. Figures are approximate and based on standard recipes."),
    ], [("Where can I see calorie counts for each product?", "menu item page")]),
    ("faq-vegan", "Vegan options", [
        ("Plant-based choices", "The Veggie Delight Burger uses a plant-based patty but the herb mayo contains egg, so it is vegetarian rather than vegan. No menu item is certified vegan."),
    ], [("Is the plant patty burger suitable for vegans?", "vegetarian rather than vegan")]),
]

POLICIES = [
    ("policy-privacy", "Privacy policy summary", [
        ("Data we keep", "We keep order history and loyalty data for 24 months after your last order. Location is only used while the app is open to find nearby stores."),
        ("Your rights", "You can export or delete your data at any time from Settings. We never sell personal data."),
    ], [("How long do you retain my purchase history after I stop ordering?", "24 months"),
        ("Do you share my details with advertisers for money?", "never sell personal data")]),
    ("policy-offer-terms", "General terms for offers", [
        ("Validity", "Offers are valid only on the dates shown and cannot be exchanged for cash. We may withdraw an offer at any time without notice."),
        ("Misuse", "Offers obtained through misuse of accounts or multiple sign-ups will be cancelled and points removed."),
    ], [("Can a coupon be swapped for money?", "cannot be exchanged for cash")]),
    ("policy-refund", "Refund policy", [
        ("Eligibility", "We refund orders that are wrong, missing items or served cold. Claims must be made within 24 hours of the order."),
        ("Method", "Refunds go to the original payment method. Gift card orders are refunded as store credit."),
    ], [("Within how long must I complain about a cold meal to get money back?", "within 24 hours"),
        ("What do I receive if I paid with a gift card and my order is refunded?", "store credit")]),
    ("policy-acceptable-use", "Acceptable use of the app", [
        ("Rules", "Do not share accounts, use automated tools to place orders or attempt to access other members' data. We may suspend accounts that break these rules."),
    ], [("Am I allowed to use bots to place orders?", "automated tools")]),
    ("policy-food-safety", "Our food safety commitment", [
        ("Standards", "All restaurants are audited by a third party twice a year and score reports are available on request. Cold food is stored below 5 degrees Celsius and cooked food is held for limited times."),
    ], [("How often are the restaurants inspected by outsiders?", "twice a year")]),
]

SOPS = [
    ("sop-ingredient-ordering", "Ingredient ordering", [
        ("Order cutoff", "Orders to the distribution center must be placed by 14:00 for next-day delivery. Review the weekly forecast every Monday."),
        ("Delivery schedule", "Frozen items such as potato and nuggets are delivered twice a week, on Tuesday and Friday. Fresh produce arrives daily before 06:00."),
    ], [("What is the deadline to submit tomorrow's supplier order?", "14:00"),
        ("On which weekdays does the frozen stock truck arrive?", "Tuesday and Friday")]),
    ("sop-drive-thru-targets", "Drive-thru service targets", [
        ("Targets", "The target average drive-thru time is 180 seconds."),
        ("Queue actions", "When the queue exceeds 10 vehicles, add one runner to take orders at the lane. Escalate to the area manager if the wait exceeds 8 minutes for 30 minutes."),
    ], [("When should an extra person go out to the lane to take orders?", "exceeds 10 vehicles"),
        ("What average service time should the drive lane achieve?", "180 seconds")]),
    ("sop-peak-staffing", "Staffing at peak", [
        ("Peak periods", "Peak periods are 8am, 12 to 1pm and 7 to 8pm. Minimum crew at peak is 7 on shift."),
        ("Using the forecast", "Use the forecast tool to adjust the roster when expected orders exceed baseline by more than 15 percent."),
    ], [("How many team members must be working during rush hours at minimum?", "7 on shift"),
        ("By how much must predicted demand exceed normal before I change the rota?", "15 percent")]),
    ("sop-food-holding", "Food safety holding times", [
        ("Hot holding", "Fries may be held for a maximum of 7 minutes in the holding bin. Cooked beef patties must be used within 20 minutes. Chicken nuggets and chicken fillets may be held for 15 minutes."),
        ("Cold storage", "Cold items must be stored below 5 degrees Celsius. Log fridge temperatures every 4 hours."),
    ], [("How long can chicken pieces sit in the warming cabinet?", "15 minutes"),
        ("What is the maximum time cooked chips can wait in the bin?", "maximum of 7 minutes"),
        ("How often must refrigerator temperatures be written down?", "every 4 hours")]),
    ("sop-waste-logging", "Waste logging", [
        ("Daily logging", "Log all discarded food in the waste tracker at close. Target waste is below 3 percent of food cost."),
        ("Forecast feedback", "Persistent over-forecasting for 2 weeks in a row should be reported so the forecast model can be retrained."),
    ], [("What percentage of food cost is the ceiling for throwaway stock?", "below 3 percent"),
        ("When do I tell head office the demand predictions are too high?", "2 weeks in a row")]),
    ("sop-refund-approvals", "Refund approvals", [
        ("Authority limits", "Crew can refund up to 5 dollars. A shift lead can approve up to 20 dollars. Above 20 dollars requires the store manager."),
    ], [("Who must sign off a 35 dollar refund?", "store manager"),
        ("What is the most a crew member may give back without asking?", "up to 5 dollars")]),
    ("sop-promotions-rules", "Promotions and campaigns", [
        ("Approved offers", "Run promotions only from the approved offer list. Do not send more than 2 promotional messages per member per week."),
    ], [("How many marketing notifications may a customer receive weekly?", "2 promotional messages")]),
    ("sop-opening-checklist", "Opening checklist", [
        ("Arrival", "The opening manager arrives 45 minutes before opening time and disables the alarm."),
        ("Equipment checks", "Heat fryers to 175 degrees Celsius, check sanitiser strength is 200 ppm and place a 150 dollar float in each till."),
    ], [("How early must the person opening the restaurant turn up?", "45 minutes before opening"),
        ("How much starting cash goes in every register?", "150 dollar float")]),
    ("sop-closing-checklist", "Closing checklist", [
        ("Tills and safe", "Close tills within 30 minutes of closing time and place cash in the safe."),
        ("Cleaning", "Deep clean the grill, drain the fryer oil into the oil tank and rotate the alarm code every month."),
    ], [("How long after shutting do registers need to be cashed up?", "within 30 minutes")]),
    ("sop-cash-handling", "Cash handling", [
        ("Safe drops", "Make a safe drop whenever a till holds more than 300 dollars. Counts are performed by two people."),
        ("Deposits and variance", "Deposit cash by 10:00 the next morning. Report any till variance above 5 dollars to the area manager."),
    ], [("At what amount in a register must money be moved to the safe?", "more than 300 dollars"),
        ("What cash discrepancy has to be escalated?", "above 5 dollars")]),
    ("sop-allergen-incident", "Allergen incident procedure", [
        ("Immediate actions", "Stop service of the affected item, change gloves and keep a sample of the food. Call emergency services immediately if the customer shows signs of anaphylaxis."),
        ("Reporting", "Report every allergen incident to the area manager within 1 hour."),
    ], [("How quickly must an allergic reaction event be reported upward?", "within 1 hour"),
        ("What do we do if a guest has trouble breathing after eating?", "emergency services")]),
    ("sop-equipment-faults", "Equipment faults and maintenance", [
        ("Priority 1 equipment", "Fryer, grill and refrigeration faults are priority 1. Call the maintenance hotline within 15 minutes of discovering the fault; response target is 4 hours."),
        ("Shake machine", "If the shake machine is down for more than 2 hours log a ticket in the facilities portal."),
    ], [("How fast must a broken freezer be reported to maintenance?", "within 15 minutes"),
        ("When do I raise a ticket for the milkshake dispenser?", "more than 2 hours")]),
    ("sop-delivery-receiving", "Receiving deliveries", [
        ("Temperature checks", "Frozen goods must be below minus 15 degrees Celsius on arrival; reject them above minus 12 degrees. Chilled goods must be below 5 degrees."),
        ("Paperwork", "Sign the delivery note within 30 minutes and report discrepancies to the supplier within 24 hours."),
    ], [("At what temperature do we refuse frozen supplies?", "above minus 12"),
        ("How long do we have to flag shortages to the vendor?", "within 24 hours")]),
    ("sop-cleaning-schedule", "Cleaning schedule", [
        ("Frequencies", "Filter fryer oil twice daily and change it every 3 days. Clean the shake machine every 14 days. Scrape the grill every hour at peak. Mop floors every 2 hours and check restrooms every 30 minutes."),
    ], [("How frequently should the cooking oil be replaced completely?", "every 3 days"),
        ("How often are the toilets inspected?", "every 30 minutes")]),
    ("sop-hygiene-uniform", "Hygiene and uniform", [
        ("Handwashing", "Wash hands at least every 30 minutes and after handling raw food, cash or waste."),
        ("Illness", "Staff with vomiting or diarrhoea must stay away from work until 48 hours after symptoms end."),
    ], [("After stomach illness, how long must a worker stay home?", "48 hours after symptoms")]),
    ("sop-breaks", "Breaks and rest periods", [
        ("Entitlement", "Shifts longer than 6 hours include a 30 minute unpaid break. Crew also get a 10 minute paid break for every 4 hours worked."),
        ("Peak cover", "No more than 2 crew may be on break at the same time during peak periods."),
    ], [("How long is the meal break on a 7 hour shift?", "30 minute unpaid break")]),
    ("sop-onboarding", "Onboarding new crew", [
        ("Training plan", "New crew shadow an experienced colleague for 3 shifts. They must complete the food safety certificate within 14 days and may work a till alone from day 5."),
    ], [("Within how many days must newcomers earn the hygiene certificate?", "within 14 days")]),
    ("sop-incident-reporting", "Incident and accident reporting", [
        ("Process", "Record injuries on form IR-12 during the same shift. Serious incidents must be phoned to the area manager within 30 minutes. Keep incident records for 3 years."),
    ], [("Which form is used to record a workplace injury?", "IR-12"),
        ("How long must accident paperwork be retained?", "3 years")]),
    ("sop-shift-handover", "Shift handover", [
        ("Handover notes", "Write handover notes in the log book covering open issues, equipment problems and stock counts for beef, chicken and potato."),
    ], [("Which stock items must be counted in the end-of-shift notes?", "beef, chicken and potato")]),
    ("sop-inventory-count", "Inventory counts", [
        ("Schedule", "Do a full count every Sunday night and a cycle count of high-value items such as beef and chicken every day."),
        ("Variance", "Investigate any variance above 2 percent and record the cause."),
    ], [("On which night is the complete stock-take performed?", "Sunday night"),
        ("What stock difference triggers an investigation?", "above 2 percent")]),
    ("sop-supplier-issues", "Supplier shortages", [
        ("Short deliveries", "For short deliveries call the distribution center within 1 hour. Emergency orders are accepted until 17:00 with a 25 dollar surcharge."),
    ], [("What is the fee for an urgent top-up order?", "25 dollar surcharge"),
        ("Until what time can an emergency order be placed?", "17:00")]),
    ("sop-power-outage", "Power outage procedure", [
        ("Service", "If power is lost for more than 30 minutes close the drive-thru and stop cooking."),
        ("Food safety", "Discard refrigerated food that has been above 5 degrees Celsius for more than 4 hours. Record all discarded items in the waste tracker."),
    ], [("How long without electricity before the lane must be closed?", "more than 30 minutes")]),
    ("sop-loyalty-fraud", "Loyalty fraud checks", [
        ("Red flags", "Flag accounts with more than 5 redemptions in one day. Verify the member's ID and raise a ticket with the loyalty team."),
    ], [("How many same-day reward redemptions look suspicious?", "more than 5 redemptions")]),
    ("sop-pest-control", "Pest control", [
        ("Inspections", "A contractor inspects every store monthly. Log sightings the same day; the contractor must attend within 24 hours of a sighting."),
    ], [("How soon must the exterminator respond after a rodent is seen?", "within 24 hours")]),
    ("sop-delivery-orders", "Delivery order handling", [
        ("Handoff", "Hand the order to the driver within 3 minutes of it being ready and seal the bag."),
        ("Late orders", "If an order arrives more than 10 minutes after the promised time, issue a 20 percent credit."),
    ], [("What compensation applies when a home delivery is over 10 minutes late?", "20 percent credit")]),
    ("sop-app-outage", "Mobile app outage", [
        ("Fallback", "If app ordering fails switch to the fallback POS code, call the technology support line and give affected customers paper vouchers."),
    ], [("What do we hand out to guests when the ordering application is down?", "paper vouchers")]),
    ("sop-lone-working", "Security and lone working", [
        ("Minimum staffing", "No one may close the store alone after 22:00; two staff must be present. Use the panic button for any threat."),
    ], [("Can one employee lock up by themselves late at night?", "two staff must be present")]),
    ("sop-complaints", "Complaint handling", [
        ("LAST method", "Use the LAST method: Listen, Apologise, Solve, Thank. Escalate to the store manager for refunds over 20 dollars or any health claim."),
        ("Logging", "Log every complaint in the CRM within 24 hours."),
    ], [("What does the acronym for handling unhappy guests stand for?", "Listen, Apologise, Solve, Thank")]),
]

TRAINING = [
    ("train-fryer", "Fryer operation", [
        ("Settings", "Set oil temperature to 175 degrees Celsius. Do not load more than 1.5 kg of fries per basket. Fries cook for 3 minutes 10 seconds."),
        ("Filtering", "Filter the oil after every 40 cooking cycles."),
    ], [("What is the maximum weight of chips per fryer basket?", "1.5 kg"),
        ("After how many cooking cycles must the oil be filtered?", "40 cooking cycles")]),
    ("train-grill", "Grill operation", [
        ("Settings", "Platen temperature is 200 degrees Celsius. A beef patty cooks for 90 seconds and must reach 71 degrees Celsius internally."),
    ], [("What core temperature must a cooked burger patty reach?", "71 degrees")]),
    ("train-shake-machine", "Soft serve and shake machine", [
        ("Daily", "Run the heat treatment cycle nightly. Keep the mix below 4 degrees Celsius."),
        ("Cleaning", "Full disassembly and cleaning is required every 14 days."),
    ], [("How cold must the ice cream mix be kept?", "below 4 degrees")]),
    ("train-coffee-machine", "Coffee machine care", [
        ("Brewing", "Brew at 93 degrees Celsius. Discard brewed coffee after 30 minutes."),
        ("Descaling", "Descale the machine every 30 days."),
    ], [("After how long should unsold brewed coffee be thrown away?", "after 30 minutes"),
        ("How regularly is limescale removed from the espresso unit?", "every 30 days")]),
    ("train-drive-thru-order-taking", "Drive-thru order taking", [
        ("Greeting", "Greet the customer within 5 seconds of the car reaching the speaker. Repeat the order back and confirm the total before the pickup window."),
    ], [("How quickly must a driver be welcomed at the speaker post?", "within 5 seconds")]),
    ("train-suggestive-selling", "Suggestive selling", [
        ("Technique", "For burger orders always offer fries and a drink. The store target for add-on attach rate is 35 percent."),
    ], [("What attach rate should each restaurant aim for on extras?", "35 percent")]),
    ("kpi-targets", "Monthly store KPIs", [
        ("Targets", "Labour cost target is 28 percent of sales and food cost 31 percent of sales. Customer satisfaction target is 85 percent and mystery shopper score must stay above 90."),
    ], [("What share of revenue should be spent on staff wages?", "28 percent of sales")]),
    ("train-forecast-tool", "Using the demand forecast and inventory plan", [
        ("Reading the forecast", "The forecast shows expected orders per day for a store. The inventory plan converts expected orders into ingredient quantities using the recipe bill of materials."),
        ("Safety stock", "Recommended order quantities include safety stock sized for a 95 percent service level. Re-run the forecast every Monday and after any local event."),
    ], [("What service level does the buffer stock in the ordering plan assume?", "95 percent service level")]),
]
