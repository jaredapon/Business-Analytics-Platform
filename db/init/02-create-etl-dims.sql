\c booklatte;

CREATE TABLE fact_transaction_dimension (
    date DATE,
    time_id VARCHAR(32),
    receipt_no INT,
    product_id VARCHAR(64),
    product_name VARCHAR(255),
    qty INT,
    price DECIMAL(10,2),
    line_total DECIMAL(12,2),
    net_total DECIMAL(12,2),
    discount DECIMAL(12,2),
    evat DECIMAL(12,2),
    pwd DECIMAL(12,2),
    senior DECIMAL(12,2),
    tax DECIMAL(12,2),
    total_gross DECIMAL(12,2),
    void INT,
    base_qty INT,
    take_out BOOLEAN,
    PRIMARY KEY (receipt_no, date),
    FOREIGN KEY (product_id) REFERENCES current_product_dimension(product_id),
    FOREIGN KEY (receipt_no) REFERENCES transaction_records(receipt_no)
    FOREIGN KEY (time_id) REFERENCES time_dimension(time_id)
);

CREATE TABLE current_product_dimension (
    product_id VARCHAR(64) PRIMARY KEY,
    product_name VARCHAR(255),
    price DECIMAL(10,2),
    last_transaction_date DATE,
    record_version INT,
    is_current BOOLEAN,
    parent_sku VARCHAR(255),
    category VARCHAR(64)
);

CREATE TABLE history_product_dimension (
    product_id VARCHAR(64),
    product_name VARCHAR(255),
    price DECIMAL(10,2),
    last_transaction_date DATE,
    record_version INT,
    is_current BOOLEAN,
    parent_sku VARCHAR(255),
    category VARCHAR(64),
    PRIMARY KEY (product_id, record_version)
);

CREATE TABLE time_dimension (
    time_id VARCHAR(32) PRIMARY KEY,
    time_desc VARCHAR(255),
    time_level INT,
    parent_id VARCHAR(32),
    FOREIGN KEY (parent_id) REFERENCES time_dimension(time_id)
);