from normalization import (
    normalize_business_name,
    normalize_address,
)


names = [
    "ABC Technologies Pvt. Ltd.",
    "ABC TECHNOLOGIES PRIVATE LIMITED",
    "abc technologies pvt ltd",
    "Orelee's Barbershop",
]

addresses = [
    "1795 Westchester Drive, High Point, NC",
    "1795 Westchester Dr., High Point, NC",
    "12 MG Road, Chennai",
]


print("=" * 70)
print("BUSINESS NAME NORMALIZATION")
print("=" * 70)

for name in names:
    print(f"Original : {name}")
    print(f"Normalized: {normalize_business_name(name)}")
    print()


print("=" * 70)
print("ADDRESS NORMALIZATION")
print("=" * 70)

for address in addresses:
    print(f"Original : {address}")
    print(f"Normalized: {normalize_address(address)}")
    print()
