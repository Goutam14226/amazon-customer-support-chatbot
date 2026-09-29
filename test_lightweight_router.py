import re


ROUTING_RULES = {

    "REFUND": [
        "refund",
        "refunded",
        "money back",
        "money hasn't come back",
        "money has not come back",
        "haven't received my money",
        "have not received my money",
        "refund not received",
        "refund not processed",
        "refund status",
        "where is my refund",
    ],

    "RETURN": [
        "return",
        "returning",
        "returned item",
        "send back",
        "send the item back",
        "return an item",
        "return this item",
    ],

    "ORDER": [
        "cancel my order",
        "cancel order",
        "cancel the order",
        "order cancellation",
        "modify my order",
        "change my order",
    ],

    "DELIVERY": [
        "delivery",
        "deliver",
        "package",
        "parcel",
        "not received",
        "not delivered",
        "hasn't arrived",
        "has not arrived",
        "didn't arrive",
        "did not arrive",
        "where is my package",
        "where is my parcel",
        "shipping",
    ],

    "PAYMENT": [
        "payment",
        "payment problem",
        "payment issue",
        "problem with my payment",
        "failed payment",
        "payment failed",
        "unable to pay",
        "can't pay",
        "cannot pay",
        "charged",
        "billing",
    ],

    "ACCOUNT": [
        "account",
        "account details",
        "account information",
        "profile",
        "personal details",
        "update my details",
        "update account",
        "change my account",
    ],
}


def route_department(query):

    q = query.lower().strip()

    scores = {}

    for department, phrases in ROUTING_RULES.items():

        score = 0

        for phrase in phrases:

            if phrase in q:
                score += 1

        scores[department] = score

    best_department = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_department]

    if best_score == 0:
        return "CONTACT", 0

    return best_department, best_score


# ============================================================
# TEST QUERIES
# ============================================================

test_queries = [

    # REFUND
    "Where is my refund?",
    "I haven't received my money yet.",
    "My refund has not been processed.",

    # RETURN
    "How do I return an item?",
    "I want to send the item back.",

    # ORDER
    "I want to cancel my order.",
    "Can I cancel the order?",

    # DELIVERY
    "I did not receive my package.",
    "My parcel hasn't arrived.",
    "Where is my package?",

    # PAYMENT
    "I have a problem with my payment.",
    "My payment failed.",
    "I cannot pay for my order.",

    # ACCOUNT
    "How can I update my account details?",
    "I want to change my personal details.",

    # UNKNOWN
    "I need help with something.",
]


print("=" * 70)
print("ROBUST LIGHTWEIGHT DEPARTMENT ROUTING TEST")
print("=" * 70)


for query in test_queries:

    department, score = route_department(query)

    print(f"\nQuery      : {query}")
    print(f"Department : {department}")
    print(f"Match score: {score}")


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)